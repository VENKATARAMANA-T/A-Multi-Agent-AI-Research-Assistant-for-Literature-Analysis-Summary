import { useEffect, useRef, useState } from 'react';
import { getAccessToken } from '../api/client';

/**
 * An image from an endpoint that requires a bearer token.
 *
 * A plain `<img src>` cannot send an Authorization header, so every figure came
 * back 401 and rendered as a broken image once the API moved behind accounts.
 * This fetches the bytes with the token and hands the element a blob URL
 * instead, which needs no credentials of its own.
 *
 * Loading is deferred until the image is near the viewport. `loading="lazy"` no
 * longer applies once the browser is not the one doing the fetching, and a
 * Figures page holding fifty-seven figures would otherwise open fifty-seven
 * requests at once and pull several megabytes nobody has scrolled to.
 */
export default function AuthImage({ src, alt, className, rootMargin = '300px' }) {
  const [objectUrl, setObjectUrl] = useState(null);
  const [state, setState] = useState('idle');
  const [visible, setVisible] = useState(false);
  const holder = useRef(null);
  const observerReported = useRef(false);

  useEffect(() => {
    const node = holder.current;
    if (!node || visible) return undefined;

    if (typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        observerReported.current = true;
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin },
    );
    observer.observe(node);

    // A working observer reports on the first frame, intersecting or not. One
    // that has said nothing at all is not working here — a hidden tab or a
    // zero-size viewport never intersects anything — and waiting on it would
    // leave an empty slot forever. Deferring is an optimisation; showing the
    // image is the job.
    const fallback = setTimeout(() => {
      if (!observerReported.current) setVisible(true);
    }, 1500);

    return () => {
      observer.disconnect();
      clearTimeout(fallback);
    };
  }, [visible, rootMargin]);

  useEffect(() => {
    if (!visible || !src) return undefined;

    let url = null;
    let cancelled = false;
    setState('loading');

    (async () => {
      try {
        const response = await fetch(src, {
          headers: { Authorization: `Bearer ${getAccessToken()}` },
        });
        if (!response.ok) throw new Error(String(response.status));

        const blob = await response.blob();
        if (cancelled) return;

        url = URL.createObjectURL(blob);
        setObjectUrl(url);
        setState('ready');
      } catch {
        if (!cancelled) setState('failed');
      }
    })();

    return () => {
      cancelled = true;
      // Released on unmount so a long session scrolling through figures does
      // not hold every blob it has ever shown.
      if (url) URL.revokeObjectURL(url);
    };
  }, [visible, src]);

  if (state === 'failed') {
    return (
      <div ref={holder} className={`image-fallback ${className || ''}`}>
        <span>Image unavailable</span>
      </div>
    );
  }

  if (state !== 'ready') {
    return <div ref={holder} className={`image-placeholder ${className || ''}`} aria-hidden="true" />;
  }

  return <img ref={holder} className={className} src={objectUrl} alt={alt} />;
}
