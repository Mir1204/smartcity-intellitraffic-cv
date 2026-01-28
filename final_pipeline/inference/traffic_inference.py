#!/usr/bin/env python3
"""
YOLOv8 Object Detection - Simple Inference Script
=================================================
Clean YOLOv8 inference using COCO pretrained model for object detection
with live preview and bounding box visualization.

Features:
- YOLOv8s COCO model (80 classes)
- Live video preview with bounding boxes
- Class names and confidence scores
- GPU acceleration with CPU fallback
- Clean, production-ready code

Author: Mir CV Engineer
Date: January 26, 2026
"""

import cv2
import torch
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO


class YOLOInference:
    """Clean YOLOv8 object detection inference with traffic class filtering, lane regions, and object tracking"""
    
    def __init__(self, model_path='../models/yolov8s.pt', confidence=0.25):
        """Initialize YOLOv8 model"""
        self.confidence = confidence
        self.frame_width = None
        self.frame_height = None
        self.lanes = {}
        
        # Historical tracking data
        self.tracking_data = []
        self.session_start = datetime.now()
        self.data_file = f"traffic_data_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Object tracking
        self.active_tracks = {}  # track_id -> {class, lane, first_seen, last_seen}
        self.total_tracks_seen = 0
        
        # Unique vehicle counting with line-crossing
        self.lane_counts = {}  # lane -> {'total': X, 'car': Y, 'bus': Z, ...}
        self.counting_lines = {}  # lane -> y_coordinate of counting line
        self.counted_tracks = {}  # lane -> set of track_ids already counted
        self.track_positions = {}  # track_id -> previous y position for line crossing detection
        
        # Traffic-relevant COCO classes (class names)
        self.traffic_classes = {
            'car', 'motorcycle', 'bus', 'truck', 'bicycle'
        }
        
        # Check device availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Loading YOLOv8 model: {model_path}")
        print(f"💻 Using device: {self.device}")
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        print(f"✅ Model loaded with {len(self.model.names)} COCO classes")
        
        # Build dynamic mapping from COCO class names to IDs for traffic classes
        self.traffic_class_ids = {}
        for class_id, class_name in self.model.names.items():
            if class_name in self.traffic_classes:
                self.traffic_class_ids[class_id] = class_name
        
        print(f"🚗 Traffic classes detected: {sorted(self.traffic_class_ids.values())}")
        print(f"🔢 Class ID mapping: {self.traffic_class_ids}")
    
    def setup_lanes(self, frame_width, frame_height):
        """Setup lane regions based on frame geometry"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define lane regions - horizontal lanes for typical traffic video
        # Bottom half of frame divided into lanes
        road_top = frame_height // 3  # Start lanes at 1/3 down
        road_bottom = frame_height
        lane_height = road_bottom - road_top
        
        # Create 3 lanes: left, center, right
        lane_width = frame_width // 3
        
        self.lanes = {
            'Left': {
                'region': (0, road_top, lane_width, road_bottom),
                'color': (255, 100, 100),  # Light blue
                'count': 0
            },
            'Center': {
                'region': (lane_width, road_top, 2 * lane_width, road_bottom),
                'color': (100, 255, 100),  # Light green  
                'count': 0
            },
            'Right': {
                'region': (2 * lane_width, road_top, frame_width, road_bottom),
                'color': (100, 100, 255),  # Light red
                'count': 0
            }
        }
        
        print(f"🛣️  Lane regions setup for {frame_width}x{frame_height} frame:")
        for lane_name, lane_info in self.lanes.items():
            x1, y1, x2, y2 = lane_info['region']
            print(f"   {lane_name}: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Setup counting lines (middle of each lane vertically)
        self.counting_lines = {}
        self.lane_counts = {}
        self.counted_tracks = {}
        
        for lane_name, lane_info in self.lanes.items():
            x1, y1, x2, y2 = lane_info['region']
            # Counting line at middle of lane (vertically)
            counting_line_y = (y1 + y2) // 2
            self.counting_lines[lane_name] = counting_line_y
            
            # Initialize counting structures
            self.lane_counts[lane_name] = {
                'total': 0,
                'car': 0,
                'motorcycle': 0,
                'bus': 0,
                'truck': 0,
                'bicycle': 0
            }
            self.counted_tracks[lane_name] = set()
        
        print(f"📏 Counting lines setup:")
        for lane_name, line_y in self.counting_lines.items():
            print(f"   {lane_name}: y={line_y}")
    
    def get_vehicle_lane(self, x1, y1, x2, y2):
        """Determine which lane a vehicle belongs to based on its bottom center point"""
        # Use bottom center of bounding box for lane assignment
        center_x = (x1 + x2) // 2
        center_y = y2  # Bottom of the box
        
        for lane_name, lane_info in self.lanes.items():
            lane_x1, lane_y1, lane_x2, lane_y2 = lane_info['region']
            if lane_x1 <= center_x <= lane_x2 and lane_y1 <= center_y <= lane_y2:
                return lane_name
        
        return None  # Vehicle outside lane regions
    
    def draw_lanes(self, frame):
        """Draw lane region boundaries on frame"""
        if not self.lanes:
            return frame
            
        for lane_name, lane_info in self.lanes.items():
            x1, y1, x2, y2 = lane_info['region']
            color = lane_info['color']
            
            # Draw semi-transparent lane regions
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
            
            # Draw lane boundaries
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add lane labels
            cv2.putText(frame, lane_name, (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def draw_counting_lines(self, frame):
        """Draw counting lines for each lane"""
        if not self.counting_lines:
            return frame
        
        for lane_name, line_y in self.counting_lines.items():
            if lane_name in self.lanes:
                lane_info = self.lanes[lane_name]
                x1, y1, x2, y2 = lane_info['region']
                color = lane_info['color']
                
                # Draw counting line across the lane width
                cv2.line(frame, (x1, line_y), (x2, line_y), color, 3)
                
                # Add small label for the counting line
                cv2.putText(frame, "COUNT", (x1 + 5, line_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def check_line_crossing(self, track_id, current_y, lane_name):
        """Check if vehicle has crossed the counting line"""
        if lane_name not in self.counting_lines or track_id in self.counted_tracks[lane_name]:
            return False  # Already counted or invalid lane
        
        counting_line_y = self.counting_lines[lane_name]
        
        # Check if we have previous position for this track
        if track_id in self.track_positions:
            previous_y = self.track_positions[track_id]
            
            # Check if vehicle crossed the line (from above to below or vice versa)
            crossed = ((previous_y <= counting_line_y <= current_y) or 
                      (current_y <= counting_line_y <= previous_y))
            
            if crossed:
                # Mark this track as counted for this lane
                self.counted_tracks[lane_name].add(track_id)
                return True
        
        # Update position for next frame
        self.track_positions[track_id] = current_y
        return False
    
    def collect_frame_data(self, frame_number, timestamp):
        """Collect and store current frame data for historical analysis"""
        # Calculate totals
        total_counted = sum(counts['total'] for counts in self.lane_counts.values())
        class_totals = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'bicycle': 0}
        for lane_counts in self.lane_counts.values():
            for vehicle_class in class_totals.keys():
                class_totals[vehicle_class] += lane_counts[vehicle_class]
        
        frame_data = {
            'timestamp': timestamp.isoformat(),
            'frame_number': frame_number,
            'active_counts': {lane: info['count'] for lane, info in self.lanes.items()},
            'cumulative_counts': {lane: counts['total'] for lane, counts in self.lane_counts.items()},
            'class_wise_counts': class_totals,
            'total_vehicles_active': sum(info['count'] for info in self.lanes.values()),
            'total_vehicles_counted': total_counted,
            'active_tracks': len(self.active_tracks),
            'total_tracks_seen': self.total_tracks_seen,
            'session_elapsed': (timestamp - self.session_start).total_seconds()
        }
        
        self.tracking_data.append(frame_data)
        
        # Save data every 30 frames (1 second at 30fps) to avoid excessive I/O
        if frame_number % 30 == 0:
            self.save_tracking_data()
        
        return frame_data
    
    def save_tracking_data(self):
        """Persist tracking data to JSON file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('../../tracking_data', exist_ok=True)
            
            data_path = f"../../tracking_data/{self.data_file}"
            
            # Include session metadata
            output_data = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'total_frames': len(self.tracking_data),
                    'frame_dimensions': f"{self.frame_width}x{self.frame_height}",
                    'lane_setup': {lane: info['region'] for lane, info in self.lanes.items()}
                },
                'tracking_data': self.tracking_data
            }
            
            with open(data_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not save tracking data: {e}")
    
    def get_traffic_trends(self):
        """Analyze recent traffic trends from collected data"""
        if len(self.tracking_data) < 60:  # Need at least 2 seconds of data
            return None
            
        # Analyze last 60 frames (2 seconds) vs previous 60 frames
        recent_data = self.tracking_data[-60:]
        previous_data = self.tracking_data[-120:-60] if len(self.tracking_data) >= 120 else []
        
        recent_avg = sum(d['total_vehicles_active'] for d in recent_data) / len(recent_data)
        
        if previous_data:
            previous_avg = sum(d['total_vehicles_active'] for d in previous_data) / len(previous_data)
            trend = "Increasing" if recent_avg > previous_avg else "Decreasing" if recent_avg < previous_avg else "Stable"
            change = ((recent_avg - previous_avg) / max(previous_avg, 1)) * 100
        else:
            trend = "Initializing"
            change = 0
            
        return {
            'trend': trend,
            'recent_avg': round(recent_avg, 1),
            'change_percent': round(change, 1)
        }
        
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame for traffic classes with lane assignment and track IDs"""
        # Reset lane counts
        for lane_info in self.lanes.values():
            lane_info['count'] = 0
        
        current_frame_tracks = set()
        detection_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
                for i, box in enumerate(boxes):
                    # Extract detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy()) if box.id[0] is not None else None
                    
                    # Filter for traffic classes only
                    if cls_id in self.traffic_class_ids and track_id is not None:
                        class_name = self.traffic_class_ids[cls_id]
                        detection_count += 1
                        
                        # Convert to int coordinates
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Determine lane
                        lane = self.get_vehicle_lane(x1, y1, x2, y2)
                        
                        # Update tracking info
                        current_frame_tracks.add(track_id)
                        if track_id not in self.active_tracks:
                            self.active_tracks[track_id] = {
                                'class': class_name,
                                'first_seen': datetime.now(),
                                'lane': lane
                            }
                            self.total_tracks_seen += 1
                        
                        # Update current lane for this track
                        self.active_tracks[track_id]['last_seen'] = datetime.now()
                        self.active_tracks[track_id]['lane'] = lane
                        
                        # Check for line crossing and count if crossed
                        center_y = (y1 + y2) // 2  # Use center Y for line crossing
                        if lane and self.check_line_crossing(track_id, center_y, lane):
                            # Vehicle crossed counting line - increment counters
                            self.lane_counts[lane]['total'] += 1
                            if class_name in self.lane_counts[lane]:
                                self.lane_counts[lane][class_name] += 1
                        
                        # Update lane count (current frame active vehicles)
                        if lane and lane in self.lanes:
                            self.lanes[lane]['count'] += 1
                            box_color = self.lanes[lane]['color']
                        else:
                            box_color = (128, 128, 128)  # Gray for vehicles outside lanes
                        
                        # Draw bounding box with lane color
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Create label with class name, confidence, track ID, and lane
                        if lane:
                            label = f"{class_name} {conf:.2f} ID:{track_id} [{lane}]"
                        else:
                            label = f"{class_name} {conf:.2f} ID:{track_id}"
                        
                        # Get text size for background rectangle
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(frame, (x1, y1 - text_height - baseline), 
                                    (x1 + text_width, y1), box_color, -1)
                        
                        # Draw text label
                        cv2.putText(frame, label, (x1, y1 - baseline), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Clean up tracks that are no longer visible (after 2 seconds)
        current_time = datetime.now()
        expired_tracks = []
        for track_id, track_info in self.active_tracks.items():
            if track_id not in current_frame_tracks:
                time_since_seen = (current_time - track_info['last_seen']).total_seconds()
                if time_since_seen > 2.0:  # Remove tracks not seen for 2 seconds
                    expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.active_tracks[track_id]
        
        # Add detection summary to frame
        cv2.putText(frame, f"Active Vehicles: {detection_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add tracking statistics
        cv2.putText(frame, f"Total Tracks: {self.total_tracks_seen} | Active: {len(self.active_tracks)}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Add cumulative lane counts
        y_offset = 130
        total_counted = sum(counts['total'] for counts in self.lane_counts.values())
        cv2.putText(frame, f"=== CUMULATIVE COUNTS ===", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Total Counted: {total_counted}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for lane_name, lane_info in self.lanes.items():
            # Current active count
            active_text = f"{lane_name}: {lane_info['count']} active"
            cv2.putText(frame, active_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, lane_info['color'], 2)
            y_offset += 20
            
            # Cumulative count
            total_count = self.lane_counts[lane_name]['total']
            cumulative_text = f"  └─ {total_count} total counted"
            cv2.putText(frame, cumulative_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        # Add class-wise totals
        y_offset += 10
        cv2.putText(frame, f"=== BY VEHICLE TYPE ===", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        class_totals = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'bicycle': 0}
        for lane_counts in self.lane_counts.values():
            for vehicle_class in class_totals.keys():
                class_totals[vehicle_class] += lane_counts[vehicle_class]
        
        for vehicle_class, count in class_totals.items():
            if count > 0:
                class_text = f"{vehicle_class.capitalize()}: {count}"
                cv2.putText(frame, class_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)
                y_offset += 25
        
        # Add traffic trends if available
        trends = self.get_traffic_trends()
        if trends:
            y_offset += 10
            trend_text = f"Trend: {trends['trend']} ({trends['change_percent']:+.1f}%)"
            trend_color = (0, 255, 0) if trends['trend'] == "Stable" else (0, 255, 255) if trends['trend'] == "Increasing" else (0, 100, 255)
            cv2.putText(frame, trend_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 2)
            y_offset += 25
            cv2.putText(frame, f"Avg: {trends['recent_avg']} vehicles", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return frame
    
    def process_video(self, video_path, save_output=None):
        """Process video with YOLOv8 detection and live preview"""
        print(f"📹 Opening video: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup lanes based on frame dimensions
        self.setup_lanes(width, height)
        
        # Setup video writer if saving output
        out_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {save_output}")
        
        frame_count = 0
        print("\n🚀 Starting inference... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                frame_count += 1
                current_time = datetime.now()
                
                # Run YOLOv8 inference with tracking enabled
                results = self.model.track(frame, conf=self.confidence, verbose=False)
                
                # Draw lane regions first (as background)
                annotated_frame = self.draw_lanes(frame)
                
                # Draw counting lines
                annotated_frame = self.draw_counting_lines(annotated_frame)
                
                # Draw detections on top of lanes
                annotated_frame = self.draw_detections(annotated_frame, results)
                
                # Collect historical data
                frame_data = self.collect_frame_data(frame_count, current_time)
                
                # Add frame counter and data collection info
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Data Points: {len(self.tracking_data)}", 
                          (10, 720-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
                
                # Save frame if output specified
                if out_writer:
                    out_writer.write(annotated_frame)
                
                # Show live preview
                cv2.imshow('YOLOv8 Object Detection', annotated_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Stopping inference (user quit)")
                    break
                
                # Progress update every second
                if frame_count % fps == 0:
                    print(f"⏳ Processing: {frame_count}/{total_frames} frames")
                    
        finally:
            # Final save of tracking data
            self.save_tracking_data()
            
            # Clean up
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            
        print(f"✅ Processing complete! Processed {frame_count} frames")
        print(f"📊 Collected {len(self.tracking_data)} data points")
        print(f"💾 Data saved to: tracking_data/{self.data_file}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Object Detection Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('video', help='Input video file path')
    parser.add_argument('--model', '-m', default='../models/yolov8s.pt',
                       help='YOLOv8 model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--output', '-o', help='Save output video path (optional)')
    
    args = parser.parse_args()
    
    # Validate input video exists
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    # Note: Model validation removed - YOLO will download if not found
    
    try:
        # Initialize inference engine
        detector = YOLOInference(args.model, args.confidence)
        
        # Process video
        detector.process_video(args.video, args.output)
        
        print("\n🎉 Detection completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Inference interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        return 1


if __name__ == "__main__":
    exit(main())