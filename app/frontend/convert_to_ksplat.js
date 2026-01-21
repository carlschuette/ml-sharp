
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import fs from 'fs';
import path from 'path';

// Polyfills for Node environment
if (typeof window === 'undefined') {
    global.window = global;
    global.self = global;
}
if (typeof File === 'undefined') {
    // Basic File polyfill if needed
    global.File = class File {
        constructor(parts, filename) {
            this.parts = parts;
            this.name = filename;
        }
    };
}
if (typeof Blob === 'undefined' && typeof window.Blob === 'undefined') {
    // Node 18+ has Blob, but if not:
    // We might need a polyfill, but let's hope it's there or not strictly needed for this path
    // SplatBuffer usually uses Uint8Array
}

const inputFile = process.argv[2];
const outputFile = process.argv[3];

if (!inputFile || !outputFile) {
    console.error("Usage: node convert_to_ksplat.js <input.ply> <output.ksplat>");
    process.exit(1);
}

try {
    const fileBuffer = fs.readFileSync(inputFile);
    // Convert node buffer to ArrayBuffer
    const arrayBuffer = fileBuffer.buffer.slice(fileBuffer.byteOffset, fileBuffer.byteOffset + fileBuffer.byteLength);

    // console.log(`Loading PLY from ${inputFile}...`);

    // Compression level 1 (16-bit)
    const compressionLevel = 1;
    const minimumAlpha = 1;
    const optimizeSplatData = true;
    const sphericalHarmonicsDegree = 0;

    GaussianSplats3D.PlyLoader.loadFromFileData(
        arrayBuffer,
        minimumAlpha,
        compressionLevel,
        optimizeSplatData,
        sphericalHarmonicsDegree
    )
        .then((splatBuffer) => {
            // console.log("Conversion complete. Saving...");
            const outBuffer = Buffer.from(splatBuffer.bufferData);
            fs.writeFileSync(outputFile, outBuffer);
            // console.log(`Saved to ${outputFile}`);
            process.exit(0);
        })
        .catch(err => {
            console.error("Conversion failed:", err);
            process.exit(1);
        });

} catch (err) {
    console.error("Error conversion execution:", err);
    process.exit(1);
}
