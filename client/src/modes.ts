import type { ProjectMode } from '../../shared/schema.ts';
import type { Tool } from './annotate/Canvas.tsx';

export type ModeConfig = {
  name: string;
  tools: Tool[];
  defaultTool: Tool;
  /** Boxes are drawn on the canvas and their coordinates listed. */
  showBoxes: boolean;
  /** Labels can define keypoints and skeletons. */
  keypoints: boolean;
  /** The auto bounding box panel is offered. */
  autoBox: boolean;
  cocoFormat: string;
  yoloFormat: string;
  emptyHint: string;
  shortcuts: [string, string][];
};

/** What each project mode shows. A new mode fails to compile until it has an entry here. */
export const MODES: Record<ProjectMode, ModeConfig> = {
  bbox: {
    name: 'Bounding box',
    tools: ['select', 'bbox'],
    defaultTool: 'bbox',
    showBoxes: true,
    keypoints: false,
    autoBox: true,
    cocoFormat: 'COCO detection JSON',
    yoloFormat: 'YOLO detection (zip)',
    emptyHint: 'Nothing annotated yet. Draw a box.',
    shortcuts: [
      ['V / B', 'Select, Box tool'],
      ['Delete', 'Delete selected box'],
      ['Arrow keys', 'Nudge selected box 1px (Shift: 10px)'],
    ],
  },
  keypoint: {
    name: 'Keypoints',
    tools: ['select', 'keypoint'],
    defaultTool: 'keypoint',
    showBoxes: false,
    keypoints: true,
    autoBox: false,
    cocoFormat: 'COCO keypoints JSON',
    yoloFormat: 'YOLO pose (zip)',
    emptyHint: 'Nothing annotated yet. Place a keypoint.',
    shortcuts: [
      ['V / K', 'Select, Keypoint tool'],
      ['Delete', 'Delete selected keypoint or instance'],
      ['Arrow keys', 'Nudge selection 1px (Shift: 10px)'],
      ['N', 'Skip to next unplaced keypoint'],
      ['O', 'Toggle keypoint visible / occluded'],
      ['Shift+click', 'Place keypoint as occluded'],
    ],
  },
};
