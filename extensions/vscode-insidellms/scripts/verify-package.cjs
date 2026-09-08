const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { entries, validate } = require('./archive.cjs');

function changeTimes(directory, date) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) changeTimes(target, date);
    fs.utimesSync(target, date, date);
  }
}

async function main() {
  const root = path.resolve(__dirname, '..');
  const manifest = require('../package.json');
  const filename = `${manifest.name}-${manifest.version}.vsix`;
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'insidellms-repro-'));
  const hashes = [];
  try {
    for (const [index, year] of [2001, 2025].entries()) {
      const build = path.join(temporary, `build-${index}`);
      fs.cpSync(root, build, { recursive: true, filter: source => {
        const relative = path.relative(root, source);
        return !relative.split(path.sep).some(part => ['node_modules', 'dist', '.editor-test'].includes(part))
          && !source.endsWith('.vsix');
      } });
      changeTimes(build, new Date(`${year}-01-01T00:00:00Z`));
      for (const args of [['ci', '--offline', '--ignore-scripts'], ['run', 'build'], ['test'], ['run', 'package']]) {
        execFileSync(process.execPath, [process.env.npm_execpath, ...args], {
          cwd: build, stdio: 'inherit', env: { ...process.env, TZ: index ? 'Pacific/Auckland' : 'UTC' }
        });
      }
      const archive = path.join(build, filename);
      validate(await entries(archive));
      hashes.push(crypto.createHash('sha256').update(fs.readFileSync(archive)).digest('hex'));
    }
    assert.equal(hashes[0], hashes[1], 'Whole VSIX archives must be byte-identical');
    console.log(JSON.stringify({ reproducible: true, sha256: hashes[0], cleanBuilds: 2 }));
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
