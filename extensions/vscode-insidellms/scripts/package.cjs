const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { normalize } = require('./archive.cjs');

async function main() {
  const root = path.resolve(__dirname, '..');
  const manifest = require('../package.json');
  const output = path.join(root, `${manifest.name}-${manifest.version}.vsix`);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'insidellms-vsix-'));
  try {
    const raw = path.join(temporary, 'raw.vsix');
    execFileSync(process.execPath, [require.resolve('@vscode/vsce/vsce'),
      'package', '--no-dependencies', '--out', raw], { cwd: root, stdio: 'inherit' });
    await normalize(raw, output);
    console.log(`Normalized local VSIX: ${output}`);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
