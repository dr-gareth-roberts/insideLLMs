const fs = require('node:fs');
const { pipeline } = require('node:stream/promises');
const yauzl = require('yauzl');
const yazl = require('yazl');

const allowed = [
  '[Content_Types].xml', 'extension.vsixmanifest', 'extension/package.json',
  'extension/dist/extension.js', 'extension/readme.md', 'extension/LICENSE.txt'
].sort();

function entries(filename) {
  return new Promise((resolve, reject) => {
    yauzl.open(filename, { lazyEntries: true }, (error, archive) => {
      if (error) return reject(error);
      const result = new Map();
      archive.on('error', reject);
      archive.on('end', () => resolve(result));
      archive.on('entry', entry => {
        if (result.has(entry.fileName)) return reject(new Error('Duplicate archive entry'));
        archive.openReadStream(entry, (streamError, stream) => {
          if (streamError) return reject(streamError);
          const chunks = [];
          stream.on('error', reject);
          stream.on('data', chunk => chunks.push(chunk));
          stream.on('end', () => {
            result.set(entry.fileName, Buffer.concat(chunks));
            archive.readEntry();
          });
        });
      });
      archive.readEntry();
    });
  });
}

function validate(content) {
  const names = [...content.keys()].sort();
  if (JSON.stringify(names) !== JSON.stringify(allowed)) {
    throw new Error(`Unexpected VSIX contents: ${names.join(', ')}`);
  }
  const manifest = JSON.parse(content.get('extension/package.json'));
  if (!content.has(`extension/${manifest.main.replace(/^\.\//, '')}`)) {
    throw new Error('Missing compiled entrypoint');
  }
}

async function normalize(source, destination) {
  const content = await entries(source);
  validate(content);
  const archive = new yazl.ZipFile();
  for (const name of [...content.keys()].sort()) {
    archive.addBuffer(content.get(name), name, {
      // ZIP stores local DOS time; construct the same wall time in every TZ.
      mtime: new Date(1980, 0, 1, 0, 0, 0), mode: 0o100644,
      compress: false, forceDosTimestamp: true
    });
  }
  archive.end();
  await pipeline(archive.outputStream, fs.createWriteStream(destination));
}

module.exports = { entries, normalize, validate, allowed };
