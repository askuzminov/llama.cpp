#!/usr/bin/env node
// Copy one tensor of another GGUF model into a copy of this one. Node 18+ or Bun, no packages.
//
// The case it was written for: the n-gram (PLE) table of Qwen3.8-Flash-Next, per_layer_token_embd.weight, is IQ4_NL
// in the UD-Q4_K_XL quant; the same table taken from the full (BF16) model gives a model whose other tensors stay
// byte for byte the same.
//
//   node gguf-replace-tensor.mjs --src <a file of the full model> --dst <a file of the quantized model> --out <dir>
//        [--tensor per_layer_token_embd.weight] [--type keep|bf16|q8_0] [--check-rows 256]
//        [--no-others] [--dry-run] [--force]
//   node gguf-replace-tensor.mjs --list <a gguf file> [name regex]
//
// Both models can be split (<name>-0000N-of-0000M.gguf): the split that holds the tensor is found in each. --out gets
// that split of --dst with the tensor replaced and, under the same names, hard links to the other splits of --dst
// (copies where a link cannot be made, as on another drive), so that --out holds a whole model; --no-others leaves
// them out. Only the data of the tensor and its type in the tensor info change: the metadata, the order and the bytes
// of every other tensor stay as they are.
//
// --type keep (default) writes the tensor in the type it has in --src. bf16 and q8_0 convert a bf16 / f16 / f32
// source as llama-quantize does (the same bytes). Before writing, --check-rows rows of the tensor in --dst and in --src
// are compared: a table with another row order shows a low correlation and stops the script (--force writes anyway).
// --dry-run stops after the checks. --list prints the tensors of a file with their types.

import fs from "node:fs";
import path from "node:path";

const GGUF_MAGIC = "GGUF";
const DEFAULT_ALIGNMENT = 32;
const CHUNK = 64 << 20;

// ggml types: id -> [name, elements per block, bytes per block]; a size of 0 is a type the script only names
const GGML_TYPES = {
    0: ["f32", 1, 4], 1: ["f16", 1, 2], 2: ["q4_0", 32, 18], 3: ["q4_1", 32, 20], 6: ["q5_0", 32, 22],
    7: ["q5_1", 32, 24], 8: ["q8_0", 32, 34], 9: ["q8_1", 32, 36], 10: ["q2_K", 256, 84], 11: ["q3_K", 256, 110],
    12: ["q4_K", 256, 144], 13: ["q5_K", 256, 176], 14: ["q6_K", 256, 210], 15: ["q8_K", 256, 292],
    16: ["iq2_xxs", 256, 66], 17: ["iq2_xs", 256, 74], 18: ["iq3_xxs", 256, 98], 19: ["iq1_s", 256, 50],
    20: ["iq4_nl", 32, 18], 21: ["iq3_s", 256, 110], 22: ["iq2_s", 256, 82], 23: ["iq4_xs", 256, 136],
    24: ["i8", 1, 1], 25: ["i16", 1, 2], 26: ["i32", 1, 4], 27: ["i64", 1, 8], 28: ["f64", 1, 8],
    29: ["iq1_m", 256, 56], 30: ["bf16", 1, 2], 34: ["tq1_0", 256, 54], 35: ["tq2_0", 256, 66],
    39: ["mxfp4", 32, 17], 40: ["nvfp4", 0, 0], 41: ["q1_0", 0, 0], 42: ["q2_0", 0, 0],
};
const T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_IQ4_NL = 20, T_BF16 = 30;
const TYPE_IDS = Object.fromEntries(Object.entries(GGML_TYPES).map(([k, v]) => [v[0], Number(k)]));

const KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

// gguf value types: id -> bytes of a scalar
const GGUF_SCALAR = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
const GGUF_STRING = 8, GGUF_ARRAY = 9;

class Fail extends Error {}

const typeName = (t) => (GGML_TYPES[t] ? GGML_TYPES[t][0] : `type ${t}`);
const typeSized = (t) => GGML_TYPES[t] && GGML_TYPES[t][2] > 0;
const align = (x, a) => Math.ceil(x / a) * a;
const gib = (n) => (n / 2 ** 30).toFixed(2);

function tensorBytes(dims, t) {
    if (!typeSized(t)) throw new Fail(`the size of ggml type ${typeName(t)} is not known to this script`);
    const [name, blck, size] = GGML_TYPES[t];
    if (dims[0] % blck) throw new Fail(`a row of ${dims[0]} elements is not a whole number of ${name} blocks`);
    return dims.reduce((a, b) => a * b, 1) / blck * size;
}

// ---- file access, positions as numbers (files far below 2^53 bytes)

function readAt(fd, pos, n) {
    const buf = Buffer.allocUnsafe(n);
    let got = 0;
    while (got < n) {
        const r = fs.readSync(fd, buf, got, n - got, pos + got);
        if (r === 0) throw new Fail("unexpected end of file");
        got += r;
    }
    return buf;
}

class Reader {
    constructor(fd) { this.fd = fd; this.pos = 0; this.buf = Buffer.alloc(0); this.bufPos = 0; }
    read(n) {
        if (this.pos < this.bufPos || this.pos + n > this.bufPos + this.buf.length) {
            this.bufPos = this.pos;
            const want = Math.max(n, 1 << 20);
            const buf = Buffer.allocUnsafe(want);
            const r = fs.readSync(this.fd, buf, 0, want, this.pos);
            if (r < n) throw new Fail("unexpected end of file in the header");
            this.buf = buf.subarray(0, r);
        }
        const o = this.pos - this.bufPos;
        this.pos += n;
        return this.buf.subarray(o, o + n);
    }
    u32() { return this.read(4).readUInt32LE(0); }
    u64() { return Number(this.read(8).readBigUInt64LE(0)); }
    string() { return this.read(this.u64()).toString("utf8"); }
    value(vtype) {
        if (vtype in GGUF_SCALAR) return this.read(GGUF_SCALAR[vtype]);
        if (vtype === GGUF_STRING) return this.string();
        if (vtype === GGUF_ARRAY) {
            const etype = this.u32();
            const n = this.u64();
            if (etype in GGUF_SCALAR) this.pos += GGUF_SCALAR[etype] * n;
            else for (let i = 0; i < n; i++) this.value(etype);
            return null;
        }
        throw new Fail(`unknown gguf value type ${vtype}`);
    }
}

// the header of a GGUF file: metadata end, alignment, tensor infos and where the data starts
function parseGguf(file) {
    const fd = fs.openSync(file, "r");
    try {
        const g = { path: file, fileSize: fs.fstatSync(fd).size, alignment: DEFAULT_ALIGNMENT, tensors: [] };
        const r = new Reader(fd);
        if (r.read(4).toString("latin1") !== GGUF_MAGIC) throw new Fail(`${file} is not a GGUF file`);
        g.version = r.u32();
        if (g.version < 2) throw new Fail(`${file}: GGUF version ${g.version} is too old`);
        const nTensors = r.u64();
        const nKv = r.u64();
        for (let i = 0; i < nKv; i++) {
            const key = r.string();
            const vtype = r.u32();
            const v = r.value(vtype);
            if (key === "general.alignment" && vtype === 4) g.alignment = v.readUInt32LE(0);
        }
        g.kvEnd = r.pos;
        for (let i = 0; i < nTensors; i++) {
            const name = r.string();
            const nDims = r.u32();
            const dims = [];
            for (let d = 0; d < nDims; d++) dims.push(r.u64());
            const type = r.u32();
            const offset = r.u64();
            g.tensors.push({ name, dims, type, offset });
        }
        g.infoEnd = r.pos;
        g.dataStart = align(g.infoEnd, g.alignment);
        // the bytes from a tensor to the next one (or to the end of the file), padding included
        const byOff = [...g.tensors].sort((a, b) => a.offset - b.offset);
        byOff.forEach((t, i) => {
            const end = i + 1 < byOff.length ? byOff[i + 1].offset : g.fileSize - g.dataStart;
            t.span = end - t.offset;
            if (t.span < 0) throw new Fail(`${file}: tensor ${t.name} lies past the end of the file`);
        });
        g.byName = new Map(g.tensors.map((t) => [t.name, t]));
        return g;
    } finally {
        fs.closeSync(fd);
    }
}

// all files of the model that a file belongs to, in split order
function splitFiles(file) {
    const m = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/.exec(file);
    if (!m) return [file];
    const n = Number(m[3]);
    const files = [];
    for (let i = 1; i <= n; i++) files.push(`${m[1]}-${String(i).padStart(5, "0")}-of-${m[3]}.gguf`);
    const missing = files.filter((f) => !fs.existsSync(f));
    if (missing.length) throw new Fail(`split files are missing: ${missing.join(", ")}`);
    return files;
}

function findTensor(file, name) {
    for (const f of splitFiles(file)) {
        const g = parseGguf(f);
        if (g.byName.has(name)) return [g, g.byName.get(name)];
    }
    throw new Fail(`no tensor ${name} in the model of ${file}`);
}

// ---- half precision

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) return s * m * 2 ** -24;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * 2 ** (e - 15);
}

// fp32 -> fp16 bits, round to nearest even (as ggml / F16C)
function floatToHalf(f) {
    f32buf[0] = f;
    const x = u32buf[0];
    const sign = (x >>> 16) & 0x8000;
    const e = (x >>> 23) & 0xff;
    let m = x & 0x7fffff;
    if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 | (m >>> 13) : 0);
    const he = e - 127 + 15;
    if (he >= 31) return sign | 0x7c00;
    if (he <= 0) {
        if (he < -10) return sign;
        m |= 0x800000;
        const shift = 14 - he;
        let hm = m >>> shift;
        const rem = m & ((1 << shift) - 1);
        const half = 1 << (shift - 1);
        if (rem > half || (rem === half && (hm & 1))) hm++;
        return sign | hm;
    }
    let h = sign | (he << 10) | (m >>> 13);
    const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
    return h;
}

// ---- reading rows for the check

function dequantRow(buf, t, n) {
    const out = new Array(n);
    if (t === T_F32) { for (let i = 0; i < n; i++) out[i] = buf.readFloatLE(i * 4); return out; }
    if (t === T_F16) { for (let i = 0; i < n; i++) out[i] = halfToFloat(buf.readUInt16LE(i * 2)); return out; }
    if (t === T_BF16) {
        for (let i = 0; i < n; i++) { u32buf[0] = buf.readUInt16LE(i * 2) << 16; out[i] = f32buf[0]; }
        return out;
    }
    if (t === T_Q8_0) {
        for (let b = 0; b < n / 32; b++) {
            const d = halfToFloat(buf.readUInt16LE(b * 34));
            for (let j = 0; j < 32; j++) out[b * 32 + j] = d * buf.readInt8(b * 34 + 2 + j);
        }
        return out;
    }
    if (t === T_Q4_0 || t === T_IQ4_NL) {
        for (let b = 0; b < n / 32; b++) {
            const d = halfToFloat(buf.readUInt16LE(b * 18));
            for (let j = 0; j < 16; j++) {
                const q = buf[b * 18 + 2 + j];
                const lo = q & 15, hi = q >> 4;
                out[b * 32 + j] = d * (t === T_Q4_0 ? lo - 8 : KVALUES_IQ4NL[lo]);
                out[b * 32 + j + 16] = d * (t === T_Q4_0 ? hi - 8 : KVALUES_IQ4NL[hi]);
            }
        }
        return out;
    }
    return null;
}

function readRows(g, t, rows) {
    const n = t.dims[0];
    const rowBytes = tensorBytes([n], t.type);
    const fd = fs.openSync(g.path, "r");
    try {
        const out = [];
        for (const r of rows) {
            const v = dequantRow(readAt(fd, g.dataStart + t.offset + r * rowBytes, rowBytes), t.type, n);
            if (!v) return null;
            out.push(v);
        }
        return out;
    } finally {
        fs.closeSync(fd);
    }
}

function correlation(a, b) {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let va = 0, vb = 0, c = 0;
    for (let i = 0; i < n; i++) {
        const x = a[i] - ma, y = b[i] - mb;
        va += x * x; vb += y * y; c += x * y;
    }
    return va === 0 || vb === 0 ? null : c / Math.sqrt(va * vb);
}

// correlation of the rows of the old and the new tensor; null when a type cannot be read here
function checkRows(srcG, srcT, dstG, dstT, nRows) {
    const total = dstT.dims.slice(1).reduce((a, b) => a * b, 1);
    let seed = 1;
    const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const set = new Set([0, total - 1]);
    while (set.size < Math.min(nRows, total)) set.add(Math.floor(rand() * total));
    const rows = [...set].sort((a, b) => a - b);
    const a = readRows(dstG, dstT, rows);
    const b = readRows(srcG, srcT, rows);
    if (!a || !b) return null;
    const cors = [];
    for (let i = 0; i < rows.length; i++) {
        const c = correlation(a[i], b[i]);
        if (c !== null) cors.push(c);
    }
    return { rows, cors };
}

// ---- conversions of a bf16 / f16 / f32 source

// typed views need an aligned offset; a buffer of a fresh allocation has one, a slice may not
function view(Ctor, buf) {
    const n = buf.length / Ctor.BYTES_PER_ELEMENT;
    if (buf.byteOffset % Ctor.BYTES_PER_ELEMENT === 0) return new Ctor(buf.buffer, buf.byteOffset, n);
    return new Ctor(Uint8Array.from(buf).buffer, 0, n);
}

function toF32(buf, t) {
    if (t === T_F32) return view(Float32Array, buf);
    const h = view(Uint16Array, buf);
    const out = new Float32Array(h.length);
    if (t === T_BF16) {
        const u = new Uint32Array(out.buffer);
        for (let i = 0; i < h.length; i++) u[i] = h[i] << 16;
    } else if (t === T_F16) {
        for (let i = 0; i < h.length; i++) out[i] = halfToFloat(h[i]);
    } else {
        throw new Fail(`only a bf16, f16 or f32 source can be converted, the source is ${typeName(t)}`);
    }
    return out;
}

function convertBf16(buf, t) {
    if (t === T_BF16) return buf;
    const u = new Uint32Array(toF32(buf, t).buffer.slice(0));
    const out = Buffer.alloc(u.length * 2);
    const h = new Uint16Array(out.buffer, out.byteOffset, u.length);
    for (let i = 0; i < u.length; i++) {
        const v = u[i];
        // ggml_compute_fp32_to_bf16: round to nearest even, a NaN stays a quiet NaN
        h[i] = (v & 0x7fffffff) > 0x7f800000 ? (v >>> 16) | 64 : Math.floor((v + 0x7fff + ((v >>> 16) & 1)) / 65536);
    }
    return out;
}

function convertQ8_0(buf, t) {
    if (t === T_Q8_0) return buf;
    const x = toF32(buf, t);
    const nb = x.length / 32;
    const out = Buffer.alloc(nb * 34);
    const dv = new DataView(out.buffer, out.byteOffset, out.length);
    const qs = new Int8Array(out.buffer, out.byteOffset, out.length);
    for (let b = 0; b < nb; b++) {
        // quantize_row_q8_0_ref: d = amax / 127, q = roundf(x * (1 / d)), all in f32
        const base = b * 32;
        let amax = 0;
        for (let j = 0; j < 32; j++) {
            const a = Math.abs(x[base + j]);
            if (a > amax) amax = a;
        }
        const d = Math.fround(amax / 127);
        const inv = d ? Math.fround(1 / d) : 0;
        const o = b * 34;
        dv.setUint16(o, floatToHalf(d), true);
        for (let j = 0; j < 32; j++) {
            const v = Math.fround(x[base + j] * inv);
            qs[o + 2 + j] = v < 0 ? -Math.floor(-v + 0.5) : Math.floor(v + 0.5);
        }
    }
    return out;
}

// ---- writing

class Progress {
    constructor(total, what) { this.total = total; this.what = what; this.done = 0; this.t0 = this.tLast = Date.now(); }
    add(n) {
        this.done += n;
        const now = Date.now();
        if (now - this.tLast >= 5000 || this.done >= this.total) {
            this.tLast = now;
            const rate = this.done / Math.max((now - this.t0) / 1000, 1e-6);
            const eta = rate > 0 ? (this.total - this.done) / rate : 0;
            console.log(`  ${this.what}: ${gib(this.done)} of ${gib(this.total)} GiB, ${(rate / 2 ** 20).toFixed(0)} MiB/s, ${eta.toFixed(0)} s left`);
        }
    }
}

function writeAll(fd, buf) {
    let off = 0;
    while (off < buf.length) off += fs.writeSync(fd, buf, off, buf.length - off);
}

function copyRange(fin, fout, start, n, progress) {
    const buf = Buffer.allocUnsafe(Math.min(CHUNK, Math.max(n, 1)));
    let pos = start;
    while (n > 0) {
        const want = Math.min(buf.length, n);
        const r = fs.readSync(fin, buf, 0, want, pos);
        if (r === 0) throw new Fail("unexpected end of file while copying");
        writeAll(fout, buf.subarray(0, r));
        pos += r;
        n -= r;
        progress.add(r);
    }
}

function freeBytes(dir) {
    try {
        const s = fs.statfsSync(dir);
        return s.bavail * s.bsize;
    } catch {
        return null; // no statfs (an older runtime): no check
    }
}

function writeSplit(dstG, srcG, srcT, name, outType, outPath) {
    const srcBytes = tensorBytes(srcT.dims, srcT.type);
    const newBytes = tensorBytes(srcT.dims, outType);
    if (srcBytes > srcT.span) throw new Fail(`${srcG.path}: ${name} should take ${srcBytes} bytes, the file has ${srcT.span}`);

    // the new layout: the tensors in the order of their data, each starting aligned
    const a = dstG.alignment;
    let cursor = 0;
    const plan = [];
    for (const t of [...dstG.tensors].sort((x, y) => x.offset - y.offset)) {
        const off = align(cursor, a);
        const size = t.name === name ? newBytes : t.span;
        plan.push({ t, off, size });
        cursor = off + size;
    }
    const newOff = new Map(plan.map((p) => [p.t.name, p.off]));

    const parts = [];
    for (const t of dstG.tensors) {
        const nb = Buffer.from(t.name, "utf8");
        const b = Buffer.alloc(8 + nb.length + 4 + 8 * t.dims.length + 4 + 8);
        let o = 0;
        b.writeBigUInt64LE(BigInt(nb.length), o); o += 8;
        nb.copy(b, o); o += nb.length;
        b.writeUInt32LE(t.dims.length, o); o += 4;
        for (const d of t.dims) { b.writeBigUInt64LE(BigInt(d), o); o += 8; }
        b.writeUInt32LE(t.name === name ? outType : t.type, o); o += 4;
        b.writeBigUInt64LE(BigInt(newOff.get(t.name)), o);
        parts.push(b);
    }
    const infos = Buffer.concat(parts);
    const headerEnd = dstG.kvEnd + infos.length;
    const dataStart = align(headerEnd, a);
    const total = dataStart + cursor;

    const free = freeBytes(path.dirname(outPath));
    if (free !== null && free < total + 2 ** 30) {
        throw new Fail(`${outPath} needs ${gib(total)} GiB, ${gib(free)} GiB are free there`);
    }

    const convert = outType === srcT.type ? null : outType === T_BF16 ? convertBf16 : convertQ8_0;
    const tmp = outPath + ".part";
    const progress = new Progress(total, path.basename(outPath));
    const fd = fs.openSync(dstG.path, "r");
    const fs_ = fs.openSync(srcG.path, "r");
    const fo = fs.openSync(tmp, "w");
    try {
        copyRange(fd, fo, 0, dstG.kvEnd, progress);
        writeAll(fo, infos);
        writeAll(fo, Buffer.alloc(dataStart - headerEnd));
        progress.add(infos.length + dataStart - headerEnd);
        let pos = dataStart;
        for (const { t, off, size } of plan) {
            const pad = dataStart + off - pos;
            if (pad) { writeAll(fo, Buffer.alloc(pad)); progress.add(pad); }
            if (t.name !== name) {
                copyRange(fd, fo, dstG.dataStart + t.offset, size, progress);
            } else if (!convert) {
                copyRange(fs_, fo, srcG.dataStart + srcT.offset, srcBytes, progress);
            } else {
                // whole source rows per chunk, each a whole number of output blocks
                const rowSrc = tensorBytes([srcT.dims[0]], srcT.type);
                const chunk = Math.max(Math.floor(CHUNK / rowSrc), 1) * rowSrc;
                let left = srcBytes, rpos = srcG.dataStart + srcT.offset, written = 0;
                while (left > 0) {
                    const n = Math.min(chunk, left);
                    const out = convert(readAt(fs_, rpos, n), srcT.type);
                    writeAll(fo, out);
                    written += out.length;
                    rpos += n;
                    left -= n;
                    progress.add(out.length);
                }
                if (written !== newBytes) throw new Fail(`wrote ${written} bytes of ${name}, expected ${newBytes}`);
            }
            pos = dataStart + off + size;
        }
        if (fs.fstatSync(fo).size !== total) throw new Fail(`wrote ${fs.fstatSync(fo).size} bytes, expected ${total}`);
    } finally {
        fs.closeSync(fo);
        fs.closeSync(fs_);
        fs.closeSync(fd);
    }
    fs.renameSync(tmp, outPath);
}

function verify(outPath, dstG, name, outType, srcG, srcT) {
    const g = parseGguf(outPath);
    if (g.tensors.map((t) => t.name).join("\n") !== dstG.tensors.map((t) => t.name).join("\n")) {
        throw new Fail(`${outPath}: the tensors differ from ${dstG.path}`);
    }
    for (const t of g.tensors) {
        const old = dstG.byName.get(t.name);
        if (t.offset % g.alignment) throw new Fail(`${outPath}: tensor ${t.name} is not aligned`);
        if (t.type !== (t.name === name ? outType : old.type) || t.dims.join() !== old.dims.join()) {
            throw new Fail(`${outPath}: tensor ${t.name} has the wrong type or shape`);
        }
        if (typeSized(t.type) && t.offset + tensorBytes(t.dims, t.type) > g.fileSize - g.dataStart) {
            throw new Fail(`${outPath}: tensor ${t.name} lies past the end of the file`);
        }
    }
    // the replaced tensor against the source: its bytes when kept, its values when converted
    const nt = g.byName.get(name);
    const n = nt.dims[0];
    const rows = nt.dims.slice(1).reduce((x, y) => x * y, 1);
    const picks = [0, Math.floor(rows / 2), rows - 1];
    const fo = fs.openSync(outPath, "r");
    const fsrc = fs.openSync(srcG.path, "r");
    try {
        const rowOut = tensorBytes([n], outType), rowSrc = tensorBytes([n], srcT.type);
        for (const r of picks) {
            const bo = readAt(fo, g.dataStart + nt.offset + r * rowOut, rowOut);
            const bs = readAt(fsrc, srcG.dataStart + srcT.offset + r * rowSrc, rowSrc);
            if (outType === srcT.type) {
                if (!bo.equals(bs)) throw new Fail(`${outPath}: row ${r} of ${name} differs from the source`);
            } else {
                const c = correlation(dequantRow(bo, outType, n), dequantRow(bs, srcT.type, n));
                if (c !== null && c < 0.999) throw new Fail(`${outPath}: row ${r} of ${name} correlates ${c.toFixed(4)} with the source`);
            }
        }
    } finally {
        fs.closeSync(fsrc);
        fs.closeSync(fo);
    }
}

function list(file, re) {
    for (const f of splitFiles(path.resolve(file))) {
        const g = parseGguf(f);
        console.log(`${path.basename(f)}: GGUF v${g.version}, ${g.tensors.length} tensors, alignment ${g.alignment}`);
        for (const t of g.tensors) {
            if (re && !re.test(t.name)) continue;
            const size = typeSized(t.type) ? `${(tensorBytes(t.dims, t.type) / 2 ** 20).toFixed(1)} MiB` : "?";
            console.log(`  ${t.name}  ${typeName(t.type)}  [${t.dims.join(", ")}]  ${size}`);
        }
    }
}

function parseArgs(argv) {
    const a = { tensor: "per_layer_token_embd.weight", type: "keep", checkRows: 256, noOthers: false, dryRun: false, force: false };
    const need = (i) => { if (i + 1 >= argv.length) throw new Fail(`${argv[i]} needs a value`); return argv[i + 1]; };
    for (let i = 0; i < argv.length; i++) {
        const k = argv[i];
        if (k === "--src") a.src = need(i++);
        else if (k === "--dst") a.dst = need(i++);
        else if (k === "--out") a.out = need(i++);
        else if (k === "--tensor") a.tensor = need(i++);
        else if (k === "--type") a.type = need(i++);
        else if (k === "--check-rows") a.checkRows = Number(need(i++));
        else if (k === "--no-others") a.noOthers = true;
        else if (k === "--dry-run") a.dryRun = true;
        else if (k === "--force") a.force = true;
        else if (k === "--list") { a.list = need(i++); if (i + 1 < argv.length) a.listRe = new RegExp(argv[++i]); }
        else if (k === "-h" || k === "--help") a.help = true;
        else throw new Fail(`unknown argument ${k}`);
    }
    return a;
}

function main() {
    const args = parseArgs(process.argv.slice(2));
    if (args.help || (!args.list && (!args.src || !args.dst || !args.out))) {
        console.log(fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").filter((l) => l.startsWith("//")).slice(0, 24).map((l) => l.slice(3)).join("\n"));
        return args.help ? 0 : 1;
    }
    if (args.list) { list(args.list, args.listRe); return 0; }
    if (!["keep", "bf16", "q8_0"].includes(args.type)) throw new Fail("--type is keep, bf16 or q8_0");

    const dstFirst = path.resolve(args.dst);
    const outDir = path.resolve(args.out);
    const same = (x, y) => (process.platform === "win32" ? x.toLowerCase() === y.toLowerCase() : x === y);
    if (same(outDir, path.dirname(dstFirst))) throw new Fail("--out must be another folder than the one of --dst");

    console.log(`finding ${args.tensor}`);
    const [dstG, dstT] = findTensor(dstFirst, args.tensor);
    const [srcG, srcT] = findTensor(path.resolve(args.src), args.tensor);
    const outType = args.type === "keep" ? srcT.type : TYPE_IDS[args.type];

    const sizeOf = (t, type) => (typeSized(type) ? `${gib(tensorBytes(t.dims, type))} GiB` : "? GiB");
    console.log(`  in --dst: ${path.basename(dstG.path)}, ${typeName(dstT.type)} [${dstT.dims.join(", ")}], ${sizeOf(dstT, dstT.type)}`);
    console.log(`  in --src: ${path.basename(srcG.path)}, ${typeName(srcT.type)} [${srcT.dims.join(", ")}], ${sizeOf(srcT, srcT.type)}`);
    console.log(`  written as ${typeName(outType)}, ${sizeOf(srcT, outType)}`);
    if (srcT.dims.join() !== dstT.dims.join()) {
        throw new Fail(`the shapes differ: [${srcT.dims}] in --src, [${dstT.dims}] in --dst`);
    }
    if (outType !== srcT.type && ![T_F32, T_F16, T_BF16].includes(srcT.type)) {
        throw new Fail(`a ${typeName(srcT.type)} source can only be kept as it is (--type keep)`);
    }

    if (args.checkRows > 0) {
        const res = checkRows(srcG, srcT, dstG, dstT, args.checkRows);
        if (!res) {
            console.log(`  row check skipped: this script cannot read ${typeName(srcT.type)} / ${typeName(dstT.type)} rows`);
        } else if (!res.cors.length) {
            console.log(`  row check: all ${res.rows.length} sampled rows are constant, nothing to compare`);
        } else {
            const c = [...res.cors].sort((x, y) => x - y);
            const mean = c.reduce((x, y) => x + y, 0) / c.length;
            console.log(`  row check, ${c.length} rows: correlation mean ${mean.toFixed(4)}, min ${c[0].toFixed(4)}, median ${c[c.length >> 1].toFixed(4)}`);
            if (mean < 0.9) {
                const msg = `the rows of --src and --dst do not match (mean correlation ${mean.toFixed(3)}): another row order?`;
                if (!args.force) throw new Fail(`${msg} --force writes anyway`);
                console.log(`  ${msg}`);
            }
        }
    }

    const outPath = path.join(outDir, path.basename(dstG.path));
    const others = splitFiles(dstFirst).filter((p) => !same(p, dstG.path));
    console.log(`writing ${outPath}${args.noOthers ? "" : `, ${others.length} other splits linked`}`);
    if (args.dryRun) { console.log("dry run, nothing written"); return 0; }

    fs.mkdirSync(outDir, { recursive: true });
    if (fs.existsSync(outPath)) throw new Fail(`${outPath} exists, remove it first`);
    const t0 = Date.now();
    writeSplit(dstG, srcG, srcT, args.tensor, outType, outPath);
    verify(outPath, dstG, args.tensor, outType, srcG, srcT);
    console.log(`  written and checked in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

    if (!args.noOthers) {
        for (const p of others) {
            const target = path.join(outDir, path.basename(p));
            if (fs.existsSync(target)) { console.log(`  ${target} exists, kept`); continue; }
            try {
                fs.linkSync(p, target);
                console.log(`  linked ${target}`);
            } catch (e) {
                console.log(`  no hard link for ${target} (${e.code || e.message}), copying ${gib(fs.statSync(p).size)} GiB`);
                fs.copyFileSync(p, target + ".part");
                fs.renameSync(target + ".part", target);
            }
        }
    }
    console.log(`done: -m ${args.noOthers ? outPath : splitFiles(outPath)[0]}`);
    return 0;
}

try {
    process.exitCode = main();
} catch (e) {
    if (e instanceof Fail) {
        console.error(`error: ${e.message}`);
        process.exitCode = 1;
    } else {
        throw e;
    }
}
