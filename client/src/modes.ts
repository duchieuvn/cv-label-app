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
  /** The auto labeling panel is offered. */
  autoBox: boolean;
  /** The panel's name, which is also its accessible region name. */
  autoName: string;
  /** What the model draws, singular and plural. */
  autoUnit: [string, string];
  /** The YOLOv8 model family it trains, as written in weight names: `yolov8n.pt` or `yolov8n-pose.pt`. */
  modelSuffix: '' | '-pose';
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
    autoName: 'Auto bounding box',
    autoUnit: ['box', 'boxes'],
    modelSuffix: '',
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
    autoBox: true,
    autoName: 'Auto keypoints',
    autoUnit: ['keypoint', 'keypoints'],
    modelSuffix: '-pose',
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
