import { memo, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { ImageMeta } from '../../../shared/schema.ts';

const ROW_HEIGHT = 26;
const OVERSCAN = 10;

type Props = {
  images: ImageMeta[];
  currentId: string;
  currentCount: number;
  hrefFor: (id: string) => string;
};

/** Renders only the rows on screen: a folder import can put 100,000 images in this list. */
export const ImageList = memo(
  function ImageList({ images, currentId, currentCount, hrefFor }: Props) {
    const navRef = useRef<HTMLElement>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [height, setHeight] = useState(800);
    const currentIndex = images.findIndex((i) => i.id === currentId);

    useLayoutEffect(() => {
      const nav = navRef.current!;
      const observer = new ResizeObserver(() => setHeight(nav.clientHeight));
      observer.observe(nav);
      return () => observer.disconnect();
    }, []);

    // Keep the current image visible as navigation moves through the list.
    useEffect(() => {
      const nav = navRef.current!;
      if (currentIndex < 0) return;
      const top = currentIndex * ROW_HEIGHT;
      if (top < nav.scrollTop) nav.scrollTop = top;
      else if (top + ROW_HEIGHT > nav.scrollTop + nav.clientHeight) nav.scrollTop = top + ROW_HEIGHT - nav.clientHeight;
    }, [currentIndex]);

    const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
    const last = Math.min(images.length, Math.ceil((scrollTop + height) / ROW_HEIGHT) + OVERSCAN);

    return (
      <nav ref={navRef} className="image-list" aria-label="Images" onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
        <div className="image-list-spacer" style={{ height: images.length * ROW_HEIGHT }}>
          {images.slice(first, last).map((image, offset) => {
            const i = first + offset;
            return (
              <a key={image.id} href={hrefFor(image.id)} className={image.id === currentId ? 'current' : ''} style={{ top: i * ROW_HEIGHT }}>
                <span className={`dot ${image.status}`} title={image.status} />
                <span className="name" title={`${image.folder}/${image.filename}`}>
                  {i + 1}. {image.filename}
                </span>
                <span className="count">{image.id === currentId ? currentCount : image.annotationCount}</span>
              </a>
            );
          })}
        </div>
      </nav>
    );
  },
  (a, b) => a.images === b.images && a.currentId === b.currentId && a.currentCount === b.currentCount,
);
