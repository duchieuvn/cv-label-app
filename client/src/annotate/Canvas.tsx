import { useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState, type PointerEvent, type Ref } from 'react';
import type { Annotation, BBox, Label } from '../../../shared/schema.ts';
import { moveAnnotation, placeKeypoint, setAnnotation, type Doc, type Selection } from './doc.ts';
import {
  clampPoint,
  fitView,
  handlePosition,
  HANDLES,
  rectFromPoints,
  resizeBox,
  screenToImage,
  zoomAt,
  type Handle,
  type Point,
  type Size,
  type View,
} from './geometry.ts';

export type Tool = 'select' | 'bbox' | 'keypoint';
export type CanvasHandle = { fit: () => void; zoom: (factor: number) => void };

type Drag =
  | { kind: 'pan'; screen: Point; view: View }
  | { kind: 'draw'; start: Point; current: Point }
  | { kind: 'move'; id: string; start: Point; current: Point; screen: Point; active: boolean }
  | { kind: 'resize'; id: string; box: BBox; handle: Handle; current: Point; screen: Point; active: boolean }
  | { kind: 'keypoint'; id: string; keypointId: string; current: Point; screen: Point; active: boolean };

/** Below this many screen pixels a press counts as a click, not a drag. */
const DRAG_THRESHOLD = 3;
const MIN_DRAWN_BOX = 2;

type Props = {
  url: string;
  size: Size;
  doc: Doc;
  labels: Label[];
  tool: Tool;
  selection: Selection;
  hidden: boolean;
  readOnly: boolean;
  /** Whether boxes are drawn. A keypoint project can hold a box from before modes existed, which is kept but not shown. */
  showBoxes: boolean;
  ref?: Ref<CanvasHandle>;
  onSelect: (selection: Selection) => void;
  onCommit: (doc: Doc) => void;
  onDrawBox: (box: BBox) => void;
  onPlace: (point: Point, occluded: boolean) => void;
  onImageError: () => void;
};

function applyDrag(doc: Doc, drag: Drag | null, size: Size): Doc {
  if (!drag) return doc;
  const update = (id: string, fn: (a: Annotation) => Annotation) => {
    const a = doc.find((x) => x.id === id);
    return a ? setAnnotation(doc, fn(a)) : doc;
  };
  switch (drag.kind) {
    case 'move':
      if (!drag.active) return doc;
      return update(drag.id, (a) => moveAnnotation(a, drag.current.x - drag.start.x, drag.current.y - drag.start.y, size));
    case 'resize':
      if (!drag.active) return doc;
      return update(drag.id, (a) => ({ ...a, bbox: resizeBox(drag.box, drag.handle, drag.current, size) }));
    case 'keypoint':
      if (!drag.active) return doc;
      return update(drag.id, (a) => placeKeypoint(a, drag.keypointId, clampPoint(drag.current, size), a.keypoints[drag.keypointId].v));
    default:
      return doc;
  }
}

export function Canvas({ url, size, doc, labels, tool, selection, hidden, readOnly, showBoxes, ref, onSelect, onCommit, onDrawBox, onPlace, onImageError }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [viewport, setViewport] = useState<Size>({ width: 0, height: 0 });
  const [view, setView] = useState<View>({ scale: 1, tx: 0, ty: 0 });
  const [drag, setDrag] = useState<Drag | null>(null);
  const [hover, setHover] = useState<Point | null>(null);
  const [spaceHeld, setSpaceHeld] = useState(false);

  const dragRef = useRef(drag);
  dragRef.current = drag;

  const fit = useCallback(() => {
    if (viewport.width > 0) setView(fitView(size, viewport));
  }, [size.width, size.height, viewport.width, viewport.height]);

  useImperativeHandle(ref, () => ({
    fit,
    zoom: (factor) => setView((v) => zoomAt(v, factor, { x: viewport.width / 2, y: viewport.height / 2 })),
  }), [fit, viewport.width, viewport.height]);

  useLayoutEffect(() => {
    const svg = svgRef.current!;
    const observer = new ResizeObserver(() => {
      const rect = svg.getBoundingClientRect();
      setViewport({ width: rect.width, height: rect.height });
    });
    observer.observe(svg);
    return () => observer.disconnect();
  }, []);

  // Fit when the image changes or the canvas first gets a size, but not on later window resizes,
  // which would throw away the user's zoom.
  const hasViewport = viewport.width > 0;
  useLayoutEffect(() => {
    if (hasViewport) setView(fitView(size, viewport));
    setDrag(null);
  }, [url, hasViewport]);

  useEffect(() => {
    const svg = svgRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const lines = e.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : 1;
      const factor = Math.exp(-e.deltaY * lines * 0.0015);
      setView((v) => zoomAt(v, factor, { x: e.clientX - rect.left, y: e.clientY - rect.top }));
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, []);

  useEffect(() => {
    const isTyping = (e: KeyboardEvent) => e.target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName);
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !isTyping(e)) {
        e.preventDefault();
        // Otherwise releasing Space would click whichever button was last used.
        if (document.activeElement instanceof HTMLButtonElement) document.activeElement.blur();
        setSpaceHeld(true);
      }
      // Runs in the capture phase so Esc cancels a drag without also deselecting.
      if (e.key === 'Escape' && dragRef.current) {
        e.stopPropagation();
        setDrag(null);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') setSpaceHeld(false);
    };
    window.addEventListener('keydown', onKeyDown, true);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown, true);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const local = (e: PointerEvent): Point => {
    const rect = svgRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: PointerEvent<SVGSVGElement>) => {
    const screen = local(e);
    const point = clampPoint(screenToImage(view, screen), size);
    const data = (e.target as SVGElement).dataset ?? {};
    svgRef.current!.setPointerCapture(e.pointerId);

    if (e.button === 1 || spaceHeld) {
      setDrag({ kind: 'pan', screen, view });
      return;
    }
    if (e.button !== 0) return;

    if (readOnly) {
      if (data.handle) return;
      if (data.ann) {
        onSelect({ annotationId: data.ann, keypointId: data.keypoint ?? null });
      } else {
        onSelect(null);
        setDrag({ kind: 'pan', screen, view });
      }
      return;
    }

    if (data.handle && selection) {
      const box = doc.find((a) => a.id === selection.annotationId)?.bbox;
      if (box) setDrag({ kind: 'resize', id: selection.annotationId, box, handle: data.handle as Handle, current: point, screen, active: false });
      return;
    }
    if (data.keypoint && data.ann) {
      onSelect({ annotationId: data.ann, keypointId: data.keypoint });
      setDrag({ kind: 'keypoint', id: data.ann, keypointId: data.keypoint, current: point, screen, active: false });
      return;
    }
    if (tool === 'bbox') {
      setDrag({ kind: 'draw', start: point, current: point });
    } else if (tool === 'keypoint') {
      onPlace(point, e.shiftKey);
    } else if (data.ann) {
      onSelect({ annotationId: data.ann, keypointId: null });
      setDrag({ kind: 'move', id: data.ann, start: point, current: point, screen, active: false });
    } else {
      onSelect(null);
      setDrag({ kind: 'pan', screen, view });
    }
  };

  const advance = (d: Drag, screen: Point): Drag => {
    if (d.kind === 'pan') return d;
    const current = screenToImage(view, screen);
    if (d.kind === 'draw') return { ...d, current: clampPoint(current, size) };
    const moved = Math.hypot(screen.x - d.screen.x, screen.y - d.screen.y) >= DRAG_THRESHOLD;
    return { ...d, current, active: d.active || moved };
  };

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const screen = local(e);
    setHover(screenToImage(view, screen));
    if (!drag) return;
    if (drag.kind === 'pan') {
      setView({
        ...drag.view,
        tx: drag.view.tx + screen.x - drag.screen.x,
        ty: drag.view.ty + screen.y - drag.screen.y,
      });
    } else {
      setDrag(advance(drag, screen));
    }
  };

  const onPointerUp = (e: PointerEvent<SVGSVGElement>) => {
    if (!drag) return;
    const final = advance(drag, local(e));
    setDrag(null);
    if (final.kind === 'draw') {
      const box = rectFromPoints(final.start, final.current);
      if (box.w >= MIN_DRAWN_BOX && box.h >= MIN_DRAWN_BOX) onDrawBox(box);
    } else {
      const next = applyDrag(doc, final, size);
      if (next !== doc) onCommit(next);
    }
  };

  const shown = useMemo(() => applyDrag(doc, drag, size), [doc, drag, size]);
  const labelOf = (a: Annotation) => labels.find((l) => l.id === a.labelId);
  const px = 1 / view.scale;

  // Largest first, so a small box inside a large one stays clickable.
  const boxes = useMemo(
    () => (showBoxes ? shown.filter((a) => a.bbox).sort((a, b) => b.bbox!.w * b.bbox!.h - a.bbox!.w * a.bbox!.h) : []),
    [shown, showBoxes],
  );
  const selected = selection ? shown.find((a) => a.id === selection.annotationId) : undefined;
  const drawing = drag?.kind === 'draw' ? rectFromPoints(drag.start, drag.current) : null;
  const cursor = drag?.kind === 'pan' ? 'grabbing' : spaceHeld ? 'grab' : tool === 'select' ? 'default' : 'crosshair';
  const showGuides = hover && !spaceHeld && tool !== 'select';
  const hoverScreen = hover && { x: hover.x * view.scale + view.tx, y: hover.y * view.scale + view.ty };
  const inImage = hover && hover.x >= 0 && hover.y >= 0 && hover.x <= size.width && hover.y <= size.height;

  return (
    <div className="canvas-wrap">
      <svg
        ref={svgRef}
        className="canvas"
        data-testid="canvas"
        data-scale={view.scale}
        data-tx={view.tx}
        data-ty={view.ty}
        style={{ cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => setDrag(null)}
        onPointerLeave={() => setHover(null)}
        onContextMenu={(e) => e.preventDefault()}
      >
        <g transform={`translate(${view.tx} ${view.ty}) scale(${view.scale})`}>
          <image
            key={url}
            href={url}
            width={size.width}
            height={size.height}
            preserveAspectRatio="none"
            onError={onImageError}
            style={{ imageRendering: view.scale >= 3 ? 'pixelated' : 'auto' }}
          />

          {!hidden && (
            <>
              {boxes.map((a) => {
                const color = labelOf(a)?.color ?? '#ffffff';
                const box = a.bbox!;
                return (
                  <rect
                    key={a.id}
                    data-ann={a.id}
                    data-testid="box"
                    x={box.x}
                    y={box.y}
                    width={box.w}
                    height={box.h}
                    fill={color}
                    fillOpacity={a.id === selection?.annotationId ? 0.22 : 0.08}
                    stroke={color}
                    strokeWidth={a.id === selection?.annotationId ? 2.5 : 1.5}
                    vectorEffect="non-scaling-stroke"
                    style={{ cursor: tool === 'select' && !drag ? 'move' : undefined }}
                  />
                );
              })}

              {boxes.map((a) => (
                <text
                  key={a.id}
                  className="box-tag"
                  x={a.bbox!.x + 3 * px}
                  y={a.bbox!.y >= 16 * px ? a.bbox!.y - 4 * px : a.bbox!.y + 13 * px}
                  fontSize={12 * px}
                  strokeWidth={3 * px}
                >
                  {labelOf(a)?.name}
                </text>
              ))}

              {shown.map((a) => {
                const label = labelOf(a);
                if (!label || label.keypoints.length === 0) return null;
                const isSelected = a.id === selection?.annotationId;
                return (
                  <g key={a.id}>
                    {label.skeleton.map(([from, to]) => {
                      const p = a.keypoints[from];
                      const q = a.keypoints[to];
                      if (!p || !q) return null;
                      return (
                        <line
                          key={`${from}-${to}`}
                          x1={p.x}
                          y1={p.y}
                          x2={q.x}
                          y2={q.y}
                          stroke={label.color}
                          strokeWidth={1.5}
                          strokeOpacity={0.8}
                          vectorEffect="non-scaling-stroke"
                          pointerEvents="none"
                        />
                      );
                    })}
                    {label.keypoints.map((k) => {
                      const p = a.keypoints[k.id];
                      if (!p) return null;
                      const active = isSelected && selection?.keypointId === k.id;
                      return (
                        <g key={k.id}>
                          <circle
                            data-testid="keypoint"
                            data-occluded={p.v === 1 ? 'true' : undefined}
                            cx={p.x}
                            cy={p.y}
                            r={(active ? 6 : 4.5) * px}
                            fill={p.v === 2 ? label.color : '#111318'}
                            stroke={active ? '#ffffff' : p.v === 2 ? '#111318' : label.color}
                            strokeWidth={1.5}
                            vectorEffect="non-scaling-stroke"
                            pointerEvents="none"
                          />
                          {isSelected && (
                            <text className="kp-tag" x={p.x + 8 * px} y={p.y + 4 * px} fontSize={11 * px} strokeWidth={3 * px}>
                              {k.name}
                            </text>
                          )}
                          {/* Larger invisible target, so points are easy to grab at any zoom. */}
                          <circle data-ann={a.id} data-keypoint={k.id} cx={p.x} cy={p.y} r={9 * px} fill="transparent" style={{ cursor: 'grab' }} />
                        </g>
                      );
                    })}
                  </g>
                );
              })}

              {showBoxes &&
                selected?.bbox &&
                HANDLES.map((handle) => {
                  const p = handlePosition(selected.bbox!, handle);
                  const s = 8 * px;
                  return (
                    <rect
                      key={handle}
                      data-handle={handle}
                      x={p.x - s / 2}
                      y={p.y - s / 2}
                      width={s}
                      height={s}
                      fill="#ffffff"
                      stroke="#111318"
                      strokeWidth={1}
                      vectorEffect="non-scaling-stroke"
                      style={{ cursor: `${handle}-resize` }}
                    />
                  );
                })}
            </>
          )}

          {drawing && (
            <rect
              x={drawing.x}
              y={drawing.y}
              width={drawing.w}
              height={drawing.h}
              fill="#ffffff"
              fillOpacity={0.1}
              stroke="#ffffff"
              strokeWidth={1.5}
              strokeDasharray="5 4"
              vectorEffect="non-scaling-stroke"
              pointerEvents="none"
            />
          )}
        </g>

        {showGuides && hoverScreen && (
          <g className="guides" pointerEvents="none">
            <line x1={hoverScreen.x} x2={hoverScreen.x} y1={0} y2={viewport.height} />
            <line x1={0} x2={viewport.width} y1={hoverScreen.y} y2={hoverScreen.y} />
          </g>
        )}
      </svg>
      <div className="canvas-readout">
        {size.width} × {size.height}
        {inImage && hover ? ` · ${Math.round(hover.x)}, ${Math.round(hover.y)}` : ''} · {Math.round(view.scale * 100)}%
      </div>
    </div>
  );
}
