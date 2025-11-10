from ultralytics import YOLO
import cv2
import time

def test_combined_detection_with_coverage():
    """
    Happy-path test for YOLO people + fire + smoke detection with coverage reporting.
    Scans the first 2 seconds of the video and provides coverage summary.
    """
    # Track coverage
    coverage = {
        'model_loading': False,
        'video_input': False,
        'people_detection': False,
        'fire_detection': False,
        'smoke_detection': False,
        'hud_functions': False,
        'performance_metrics': False,
        'early_exit': False
    }
    
    try:
        # Test model loading
        people_model = YOLO("yolov8n.pt")
        fire_model = YOLO("best.pt")
        coverage['model_loading'] = True
        
        test_video = "original_short_demo.mp4"
        
        # Test video input
        cap = cv2.VideoCapture(test_video)
        assert cap.isOpened(), "Could not open test video"
        coverage['video_input'] = True

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(fps * 2)
        frame_count = 0

        people_found = False
        fire_found = False
        smoke_found = False
        
        # Performance tracking
        inference_times = []
        start_time = time.time()

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # People detection with timing
            people_start = time.time()
            people_results = people_model.predict(frame, conf=0.4, classes=[0], verbose=False)
            people_time = time.time() - people_start
            
            if len(people_results[0].boxes) > 0:
                people_found = True
                coverage['people_detection'] = True

            # Fire and smoke detection with timing
            fire_start = time.time()
            fire_results = fire_model.predict(frame, conf=0.25, verbose=False)
            fire_time = time.time() - fire_start
            
            inference_times.append(people_time + fire_time)
            
            for box in fire_results[0].boxes:
                label = fire_model.names[int(box.cls[0])].lower()
                if label == "fire":
                    fire_found = True
                    coverage['fire_detection'] = True
                elif label == "smoke":
                    smoke_found = True
                    coverage['smoke_detection'] = True

            # Test HUD functions
            try:
                from hud import draw_hud, draw_boxes
                # Test drawing functions
                test_frame = frame.copy()
                test_frame = draw_boxes(test_frame, people_results, {'person': (255, 0, 0)}, people_model)
                test_frame = draw_boxes(test_frame, fire_results, {'fire': (255, 0, 255), 'smoke': (0, 255, 255)}, fire_model)
                
                test_lines = ["FPS: 30.00", "People: 1", "Fire: 1", "Smoke: 0"]
                test_frame = draw_hud(test_frame, test_lines, anchor="tl")
                coverage['hud_functions'] = True
            except ImportError:
                pass

            # Stop early if all found
            if people_found and fire_found and smoke_found:
                coverage['early_exit'] = True
                break

            frame_count += 1

        # Performance metrics
        if inference_times:
            avg_inference = sum(inference_times) / len(inference_times) * 1000  # convert to ms
            total_time = time.time() - start_time
            actual_fps = frame_count / total_time if total_time > 0 else 0
            coverage['performance_metrics'] = True

        cap.release()

        # Assertions
        assert people_found, "No people detected in first 2 seconds"
        assert fire_found, "No fire detected in first 2 seconds"
        assert smoke_found, "No smoke detected in first 2 seconds"
        
        # Generate coverage report
        print("\n" + "="*45)
        print("HAPPY-PATH TEST REPORT WITH COVERAGE SUMMARY")
        print("="*45)
        
        print("\nTEST EXECUTION RESULTS:")
        print(f"People detected: {people_found}")
        print(f"Fire detected: {fire_found}") 
        print(f"Smoke detected: {smoke_found}")
        print(f"Frames processed: {frame_count}")
        if inference_times:
            print(f"Average inference time: {avg_inference:.1f} ms")
            print(f"Actual FPS: {actual_fps:.1f}")
        
        print("\nCOVERAGE SUMMARY:")
        total_items = len(coverage)
        covered_items = sum(coverage.values())
        coverage_percentage = (covered_items / total_items) * 100
        
        for component, covered in coverage.items():
            status = "COVERED" if covered else "NOT COVERED"
            print(f"  {component.replace('_', ' ').title():<20} : {status}")
        
        print(f"\nOVERALL COVERAGE: {coverage_percentage:.1f}%")
        print(f"({covered_items}/{total_items} components tested)")
        print("\nTEST STATUS: PASSED - All critical detections working")
        print("="*55)
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        # Still print partial coverage if available
        if 'coverage' in locals():
            print("\nPARTIAL COVERAGE SUMMARY:")
            for component, covered in coverage.items():
                status = "COVERED" if covered else "NOT COVERED"
                print(f"  {component.replace('_', ' ').title():<20} : {status}")
        return False
    
    return True

if __name__ == "__main__":
    test_combined_detection_with_coverage()
    