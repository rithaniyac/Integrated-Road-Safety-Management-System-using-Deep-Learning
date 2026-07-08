import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from datetime import datetime

class TrafficViolationDetector:
    def __init__(self):
        # Initialize YOLO models
        self.helmet_model = YOLO('yolov8n.pt')  # For person/motorcycle detection
        self.reader = easyocr.Reader(['en'])
        
        # Define speed threshold (km/h)
        self.speed_threshold = 60
        
        # For speed calculation
        self.prev_positions = {}
        self.fps = 30  # Camera FPS
        self.pixel_to_meter = 0.05  # Calibration factor
        
    def detect_helmet(self, frame, person_box):
        """Detect if person is wearing helmet"""
        x1, y1, x2, y2 = map(int, person_box)
        person_roi = frame[y1:y2, x1:x2]
        
        # Simple color-based helmet detection
        # In production, use a trained helmet detection model
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
        
        # Detect head region (upper 1/3 of person)
        head_height = int((y2 - y1) * 0.33)
        head_roi = person_roi[0:head_height, :]
        
        # Check for helmet-like colors/features
        # This is simplified - use proper ML model in production
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        helmet_detected = np.mean(gray) < 100  # Dark object on head
        
        return helmet_detected
    
    def count_riders(self, frame, motorcycle_box):
        """Count number of riders on motorcycle"""
        x1, y1, x2, y2 = map(int, motorcycle_box)
        bike_roi = frame[y1:y2, x1:x2]
        
        # Detect persons in motorcycle region
        results = self.helmet_model(bike_roi, classes=[0])  # class 0 = person
        rider_count = len(results[0].boxes)
        
        return rider_count
    
    def detect_number_plate(self, frame, vehicle_box):
        """Extract number plate using OCR"""
        x1, y1, x2, y2 = map(int, vehicle_box)
        vehicle_roi = frame[y1:y2, x1:x2]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Detect text
        results = self.reader.readtext(gray)
        
        plates = []
        for (bbox, text, prob) in results:
            if prob > 0.5 and len(text) > 5:  # Filter noise
                plates.append(text.replace(" ", ""))
        
        return plates[0] if plates else None
    
    def calculate_speed(self, object_id, current_pos):
        """Calculate speed based on position change"""
        if object_id in self.prev_positions:
            prev_pos = self.prev_positions[object_id]
            
            # Calculate pixel distance
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            pixel_distance = np.sqrt(dx**2 + dy**2)
            
            # Convert to real distance (meters)
            distance = pixel_distance * self.pixel_to_meter
            
            # Calculate speed (m/s to km/h)
            time_elapsed = 1 / self.fps
            speed_mps = distance / time_elapsed
            speed_kmh = speed_mps * 3.6
            
            self.prev_positions[object_id] = current_pos
            return speed_kmh
        else:
            self.prev_positions[object_id] = current_pos
            return 0
    
    def process_frame(self, frame):
        """Process single frame for violations"""
        violations = []
        
        # Detect motorcycles
        results = self.helmet_model(frame, classes=[3])  # class 3 = motorcycle
        
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Calculate speed
            speed = self.calculate_speed(idx, center)
            
            # Count riders
            rider_count = self.count_riders(frame, [x1, y1, x2, y2])
            
            # Check for violations
            violation_data = {
                'timestamp': datetime.now().isoformat(),
                'violations': [],
                'number_plate': None
            }
            
            # Check helmet
            helmet_worn = self.detect_helmet(frame, [x1, y1, x2, y2])
            if not helmet_worn:
                violation_data['violations'].append('NO_HELMET')
            
            # Check speed
            if speed > self.speed_threshold:
                violation_data['violations'].append(f'OVERSPEEDING_{int(speed)}kmh')
            
            # Check triple riding
            if rider_count >= 3:
                violation_data['violations'].append(f'TRIPLE_RIDING_{rider_count}_riders')
            
            # Extract number plate if any violation
            if violation_data['violations']:
                plate = self.detect_number_plate(frame, [x1, y1, x2, y2])
                violation_data['number_plate'] = plate
                violations.append(violation_data)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Violation: {', '.join(violation_data['violations'])}", 
                           (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, violations

# Usage Example
def main():
    detector = TrafficViolationDetector()
    cap = cv2.VideoCapture('input.mp4')  # or 0 for webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, violations = detector.process_frame(frame)
        
        # Log violations
        for violation in violations:
            print(f"Violation detected: {violation}")
            # Here you could save to database or send alerts
        
        cv2.imshow('Traffic Monitoring', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
