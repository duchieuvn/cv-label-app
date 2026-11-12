import { open, readFile } from 'node:fs/promises';
import { imageSize } from 'image-size';

const TYPES = {
  jpg: { extension: '.jpg', mime: 'image/jpeg' },
  png: { extension: '.png', mime: 'image/png' },
  webp: { extension: '.webp', mime: 'image/webp' },
  gif: { extension: '.gif', mime: 'image/gif' },
  bmp: { extension: '.bmp', mime: 'image/bmp' },
};
type ImageType = keyof typeof TYPES;

/** Most dimensions sit near the start of a file, so scans avoid reading whole images. */
const HEADER_BYTES = 256 * 1024;

function sniff(b: Uint8Array): ImageType | null {
  const ascii = (start: number, end: number) => String.fromCharCode(...b.subarray(start, end));
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'jpg';
  if (b[0] === 0x89 && ascii(1, 4) === 'PNG') return 'png';
  if (ascii(0, 4) === 'GIF8') return 'gif';
  if (ascii(0, 2) === 'BM') return 'bmp';
  if (ascii(0, 4) === 'RIFF' && ascii(8, 12) === 'WEBP') return 'webp';
  return null;
}

/** Display dimensions, swapped when EXIF rotates the image by 90 degrees. */
function dimensions(buffer: Uint8Array) {
  const info = imageSize(buffer);
  if (!info.width || !info.height) throw new Error('image has no dimensions');
  const rotated = (info.orientation ?? 1) >= 5;
  return { width: rotated ? info.height : info.width, height: rotated ? info.width : info.height };
}

export type ImageInfo = { extension: string; mime: string; width: number; height: number };

/**
 * Identifies a file from its contents, never its name. Returns null if it is not a supported image,
 * and throws if it looks like one but cannot be read or parsed.
 */
export async function inspectImage(path: string): Promise<ImageInfo | null> {
  const file = await open(path, 'r');
  let header: Uint8Array;
  let size: number;
  try {
    size = (await file.stat()).size;
    const buffer = Buffer.alloc(Math.min(HEADER_BYTES, size));
    const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
    header = buffer.subarray(0, bytesRead);
  } finally {
    await file.close();
  }
  const type = sniff(header);
  if (!type) return null;
  try {
    return { ...TYPES[type], ...dimensions(header) };
  } catch (error) {
    // A large EXIF block can push a JPEG's dimensions past the header.
    if (size <= HEADER_BYTES) throw error;
    return { ...TYPES[type], ...dimensions(await readFile(path)) };
  }
}
