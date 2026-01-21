const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

const rootDir = path.resolve(__dirname, '..');
const isWin = os.platform() === 'win32';

// Determine the Python executable path based on OS
const pythonPath = isWin
    ? path.join(rootDir, '.venv', 'Scripts', 'python.exe')
    : path.join(rootDir, '.venv', 'bin', 'python');

const downloadScript = path.join(rootDir, 'app', 'backend', 'download_models.py');
const mainScript = path.join(rootDir, 'app', 'backend', 'main.py');

function runPython(scriptPath) {
    return new Promise((resolve, reject) => {
        console.log(`[Backend Runner] Executing: ${pythonPath} "${scriptPath}"`);
        // Use shell: true to help with path resolution on some systems, though spawn usually handles exe paths fine
        const proc = spawn(pythonPath, [scriptPath], { stdio: 'inherit' });

        proc.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`Script ${scriptPath} exited with code ${code}`));
        });

        proc.on('error', (err) => {
            reject(err);
        });
    });
}

(async () => {
    try {
        console.log('[Backend Runner] --- Checking/Downloading Models ---');
        await runPython(downloadScript);

        console.log('[Backend Runner] --- Starting Backend ---');
        await runPython(mainScript);
    } catch (err) {
        console.error('[Backend Runner] Error:', err.message);
        console.error('[Backend Runner] Ensure you have run the installation script and that .venv exists.');
        process.exit(1);
    }
})();
