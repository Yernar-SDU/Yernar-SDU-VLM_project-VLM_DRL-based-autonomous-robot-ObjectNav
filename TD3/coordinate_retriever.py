#!/usr/bin/env python3
"""
Object Coordinate Retriever with Navigation Performance Metrics
Queries a JSONL database of detected objects and tracks SR/SPL metrics
"""
import torch

import json
import re
import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from robot_runner import RobotRunner
from sota_runner import SOTARunner
import math
import pandas as pd

class NavigationMetrics:
    """Track aggregated navigation performance across trials"""

    def __init__(self):
        self.episodes = []
        self.episode_summaries = []
        self.attempts = []


    def add_attempt(self, attempt_data: Dict):
        """
        Save single navigation attempt (not averaged)
        """
        attempt_data['timestamp'] = time.time()
        self.attempts.append(attempt_data)



    def compute_metrics(self) -> Dict:
        """Compute overall averages (SR and SPL) across all recorded goals."""
        if not self.episodes:
            return {
                'avg_success_rate': 0.0,
                'avg_path_length': 0.0,
                'avg_shortest_path': 0.0,
                'avg_spl': 0.0,
                'prediction_error': 0.0,
                'total_episodes': 0
            }

        return {
            'avg_success_rate': np.mean([ep['success_rate'] for ep in self.episodes]),
            'avg_path_length': np.mean([ep['avg_path_length'] for ep in self.episodes]),
            'avg_shortest_path': np.mean([ep['avg_shortest_path'] for ep in self.episodes]),
            'avg_spl': np.mean([ep['avg_spl'] for ep in self.episodes]),
            'prediction_error': np.mean([ep['prediction_error'] for ep in self.episodes]),
            'total_episodes': len(self.episodes)
        }



    def print_metrics(self):
        """Display a formatted summary of navigation performance"""
        m = self.compute_metrics()

        print("\n" + "="*70)
        print("📊 NAVIGATION PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Success Rate (SR):     {m['avg_success_rate']:.3f} ({m['avg_success_rate']*100:.1f}%)")

        # Compute SPL average safely
        spl_values = [ep.get('avg_spl', 0.0) for ep in self.episodes if 'avg_spl' in ep]
        avg_spl = np.mean(spl_values) if spl_values else 0.0
        print(f"Average SPL:           {avg_spl:.3f}")
        print(f"Prediction error:      {m['prediction_error']:.3f}")
        print(f"Total Goals Tested:    {m['total_episodes']}")
        print("="*70 + "\n`")


    def save_to_file(self, filename: str = "navigation_metrics_MNDR.json"):
        """Save metrics to JSON"""
        data = {
            'summary': self.compute_metrics(),
            'episodes': self.episodes
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Metrics saved to {filename}")

    def reset(self):
        self.episodes = []

    def save_attempts_to_excel(self, filename="navigation_attempts_signs.xlsx"):
        if not self.attempts:
            print("⚠ No attempts to save.")
            return

        df = pd.DataFrame(self.attempts)
        df.to_excel(filename, index=False)
        print(f"💾 Saved {len(self.attempts)} attempts.")



class ObjectCoordinateRetriever:
    def __init__(self, objects_file: str, api_key: Optional[str] = None):
        """
        Initialize the retriever
        
        Args:
            objects_file: Path to detected_objects.jsonl
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.objects_file = objects_file
        self.objects_cache = []
        self.client = OpenAI(api_key="REMOVED_API_KEY")
        
        # Navigation metrics tracker
        self.metrics = NavigationMetrics()
        
        # Current episode tracking
        self.current_episode = {
            'start_x': 0.0,
            'start_y': 0.0,
            'goal_x': None,
            'goal_y': None,
            'path_traveled': 0.0,
            'last_x': 0.0,
            'last_y': 0.0
        }
        
        # Load objects database
        self.reload_objects()

    
    def reload_objects(self):
        """Reload objects from JSONL file"""
        self.objects_cache = []
        
        try:
            with open(self.objects_file, 'r') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        self.objects_cache.append(obj)
            print(f"✅ Loaded {len(self.objects_cache)} objects from database")
        except FileNotFoundError:
            print(f"❌ Objects file not found: {self.objects_file}")
        except Exception as e:
            print(f"❌ Error loading objects: {e}")
    
    def get_available_objects(self) -> List[str]:
        """Get list of unique object classes"""
        return sorted(list(set(obj['object_class'] for obj in self.objects_cache)))
    
    def start_navigation_episode(self, start_x: float, start_y: float, 
                                  goal_x: float, goal_y: float):
        """
        Start tracking a new navigation episode
        
        Args:
            start_x, start_y: Starting position
            goal_x, goal_y: Goal position
        """
        self.current_episode = {
            'start_x': start_x,
            'start_y': start_y,
            'goal_x': goal_x,
            'goal_y': goal_y,
            'path_traveled': 0.0,
            'last_x': start_x,
            'last_y': start_y
        }

    def record_failed_episode(self):
            """
            Record a failed navigation attempt (object not found / not detected)
            """
            self.metrics.add_attempt({
                'success': 0.0,
                'avg_path_length': 0.0,
                'avg_shortest_path': 0.0,
                'avg_spl': 0.0,
                'prediction_error': 0.0,
                'taken_time': 0.0,
            })

    
    def update_position(self, current_x: float, current_y: float):
        """
        Update robot position and accumulate path length
        
        Args:
            current_x, current_y: Current robot position
        """
        # Calculate distance traveled since last update
        dx = current_x - self.current_episode['last_x']
        dy = current_y - self.current_episode['last_y']
        distance = math.sqrt(dx**2 + dy**2)
        
        self.current_episode['path_traveled'] += distance
        self.current_episode['last_x'] = current_x
        self.current_episode['last_y'] = current_y
    
    def end_navigation_episode(self, success: bool, 
                               final_x: float, final_y: float):
        """
        End navigation episode and record metrics
        
        Args:
            success: Whether robot reached the goal
            final_x, final_y: Final robot position
        """
        # Update to final position
        self.update_position(final_x, final_y)
        
        # Calculate shortest path length (straight-line distance)
        dx = abs(self.current_episode['goal_x'] - self.current_episode['start_x'])
        dy = abs(self.current_episode['goal_y'] - self.current_episode['start_y'])

        # Old: shortest_path = math.sqrt(dx**2 + dy**2)

        # New: This better reflects the 'Optimal Path' in your Gazebo room
        euclidean = math.sqrt(dx**2 + dy**2)
       
        # This ensures the reference distance acknowledges the need to navigate around walls
        shortest_path = euclidean
        
        
        print(f"\n📈 Episode completed:")
        print(f"   Success: {'✅ Yes' if success else '❌ No'}")
        print(f"   Path traveled: {self.current_episode['path_traveled']:.2f} m")
        print(f"   Shortest path: {shortest_path:.2f} m")
        if success and shortest_path > 0:
            efficiency = shortest_path / self.current_episode['path_traveled']
            print(f"   Efficiency: {efficiency:.2f}")
    
    def extract_object_query(self, user_question: str) -> Optional[str]:
        """
        Use GPT to extract the object the user is asking about
        
        Returns: Standardized object name or None
        """
        available_objects = self.get_available_objects()
        
        prompt = f"""
            You are extracting an object name from a user query.

            Available detected objects:
            {', '.join(available_objects)}

            User query:
            "{user_question}"

            TASK:
            Return the BEST matching object_class from the list.

            IMPORTANT RULES:
            - Return ONLY ONE object_class from the list
            - NEVER return NONE unless absolutely no reasonable match exists
            - Match by MEANING, not exact wording """
        # Examples:
        # - "Where is the box?" → "blue_box" or "brick_box" (pick most common)
        # - "Find me blue box" → "blue_box"
        # - "go to the hydrant" → "red_fire_hydrant"
        # - "can you go to fire hydrant?" → "red_fire_hydrant"
        # - "find the brick box" → "brick_box"
        # - "where is the wall" → "brick_wall"

        # Return ONLY the object name, nothing else.

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract object names from questions. Return only the object name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            object_name = response.choices[0].message.content.strip()
            
            # Validate it's in our list
            if object_name in available_objects:
                return object_name
            elif object_name == "NONE":
                return None
            else:
                # Try fuzzy matching
                for obj in available_objects:
                    if object_name.lower() in obj.lower() or obj.lower() in object_name.lower():
                        return obj
                return None
                
        except Exception as e:
            print(f"❌ GPT extraction error: {e}")
            return None
    
    def find_object_coordinates(self, object_class: str) -> List[Dict]:
        """
        Find all instances of an object class
        
        Returns: List of objects with their coordinates
        """
        matches = [obj for obj in self.objects_cache 
                   if obj['object_class'] == object_class]
        return matches
    
    def get_closest_object(self, object_class: str, 
                          robot_x: float = 0.0, 
                          robot_y: float = 0.0) -> Optional[Dict]:
        """
        Get the closest instance of an object to a given position
        
        Args:
            object_class: Object to find
            robot_x, robot_y: Reference position (default: origin)
        
        Returns: Closest object or None
        """
        matches = self.find_object_coordinates(object_class)
        
        if not matches:
            return None
        
        # Calculate distances and find closest
        closest = None
        min_dist = float('inf')
        
        for obj in matches:
            dx = obj['world_x'] - robot_x
            dy = obj['world_y'] - robot_y
            dist = (dx**2 + dy**2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                closest = obj
        
        return closest
    
    def query(self, user_question: str, 
              robot_x: float = 0.0, 
              robot_y: float = 0.0) -> Dict:
        """
        Main query interface - answer user questions about object locations
        
        Args:
            user_question: Natural language question
            robot_x, robot_y: Current robot position (for "closest" logic)
        
        Returns: Dict with object info and coordinates
        """
        # Extract object from question
        object_name = self.extract_object_query(user_question)
        
        if not object_name:
            self.record_failed_episode()
            return {
                "success": False,
                "error": "Could not understand which object you're asking about",
                "available_objects": self.get_available_objects()
            }

        # Find the object
        matches = self.find_object_coordinates(object_name)
        
        if not matches:
            self.record_failed_episode()
            return {
                "success": False,
                "error": f"No '{object_name}' found in database",
                "object_queried": object_name
            }

        
        # Get closest instance
        closest = self.get_closest_object(object_name, robot_x, robot_y)
        
        # Calculate distance
        dx = abs(closest['world_x'] - robot_x)
        dy = abs(closest['world_y'] - robot_y)
        distance = (dx**2 + dy**2) ** 0.5
        # distance = max((dx**2 + dy**2)**0.5, (dx + dy) / 1.2)
        
        return {
            "success": True,
            "object_class": object_name,
            "coordinates": {
                "x": closest['world_x'],
                "y": closest['world_y'],
                "z": closest['world_z']
            },
            "distance_from_robot": distance,
            "detection_info": {
                "confidence": closest['confidence'],
                "method": closest['detection_method'],
                "timestamp": closest['timestamp'],
                "observation_count": closest['observation_count']
            },
            "total_instances": len(matches),
            "all_instances": matches if len(matches) > 1 else None
        }
    
    def format_response(self, result: Dict) -> str:
        """Format query result as human-readable text"""
        if not result['success']:
            error_msg = result['error']
            if 'available_objects' in result:
                objs = ', '.join(result['available_objects'])
                return f"❌ {error_msg}\n\n📋 Available objects: {objs}"
            return f"❌ {error_msg}"
        
        obj = result['object_class']
        coords = result['coordinates']
        dist = result['distance_from_robot']
        count = result['total_instances']
        
        response = f"""✅ Found {obj}!

            📍 Coordinates:
            X: {coords['x']:.2f} m
            Y: {coords['y']:.2f} m
            Z: {coords['z']:.2f} m

            📏 Distance: {dist:.2f} m from current position

            ℹ️  Detection info:
            • Confidence: {result['detection_info']['confidence']*100:.0f}%
            • Method: {result['detection_info']['method'].upper()}
            • Seen {result['detection_info']['observation_count']} time(s)"""
        
        if count > 1:
            response += f"\n   • Total instances found: {count}"
        
        return response
    


def main():
    """Interactive command-line interface"""
    import sys
    
    # Configuration
    OBJECTS_FILE = "/tmp/exploration_data/detected_objects.jsonl"
    robotRunner = RobotRunner()
    # sotaRunner = SOTARunner()
    print("="*70)
    print("🤖 OBJECT COORDINATE RETRIEVER with SR/SPL Metrics")
    print("="*70)
    print(f"Database: {OBJECTS_FILE}")
    print()
    
    # Initialize
    retriever = ObjectCoordinateRetriever(OBJECTS_FILE)
    
    if not retriever.objects_cache:
        print("❌ No objects loaded. Please check the file path.")
        return
    
    print(f"✅ Loaded {len(retriever.objects_cache)} objects")
    print(f"📋 Available: {', '.join(retriever.get_available_objects())}")
    print()
    print("="*70)
    print("Commands:")
    print("  • Ask: 'Where is the box?', 'Find blue box', etc.")
    print("  • 'metrics' - Show SR/SPL performance")
    print("  • 'save' - Save metrics to file")
    print("  • 'reset' - Clear metrics")
    print("  • 'reload' - Refresh object database")
    print("  • 'list' - Show available objects")
    print("  • 'quit' - Exit")
    print("="*70)
    print()
    
    # Interactive loop
    # Interactive loop in main()
    QUESTION_FILE = "questions.txt" 
    # Try to load questions from file (optional)
    if os.path.exists(QUESTION_FILE):
        with open(QUESTION_FILE, "r", encoding="utf-8") as f:
            question_list = [line.strip() for line in f if line.strip()]
        question_iter = iter(question_list)
        print(f"📂 Loaded {len(question_list)} questions from '{QUESTION_FILE}'")
    else:
        question_iter = None
    while True:
        try:
            if question_iter:
                try:
                    question = next(question_iter)
                    print(f"\n🔍 Question (from file): {question}")
                except StopIteration:
                    print("\n✅ Finished all file questions. Switching to interactive mode.\n")
                    question_iter = None
                    continue
            else:
                question = input("🔍 Question: ").strip()
                
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'reload':
                retriever.reload_objects()
                continue
            
            if question.lower() == 'list':
                print(f"📋 Available objects: {', '.join(retriever.get_available_objects())}")
                continue
            
            if question.lower() == 'metrics':
                retriever.metrics.print_metrics()
                continue
            
            if question.lower() == 'save':
                retriever.metrics.save_to_file()
                retriever.metrics.save_attempts_to_excel()
                continue
            
            if question.lower() == 'reset':
                retriever.metrics.reset()
                print("✅ Metrics reset")
                continue
            
            # Query the database
            print()
            result = retriever.query(question, robot_x=0.0, robot_y=0.0)
            # result_ground_y = dict_quest_coord['question']
            # result_ground_x = dict_quest_coord['question']

            print(retriever.format_response(result) + "\n")
            
            # If successful, navigate and track metrics
            if result['success']:
                coords = result['coordinates']
                goal_x, goal_y = coords['x'], coords['y']

                object_name = result['object_class']
                nav_results = None
                true_x, true_y = 0, 0
                # 
                ground_true_table = pd.read_csv('_models.csv')
                coordinates = ground_true_table.loc[ground_true_table['Model Name'] == object_name, ['X', 'Y', 'Z']]
                if not coordinates['X'].empty:
                    true_x, true_y = coordinates['X'].iloc[0], coordinates['Y'].iloc[0]
                    torch.cuda.empty_cache()

                    nav_results = robotRunner.sendGoal(goal_x, goal_y, true_x, true_y, object_name)
                    # nav_results = sotaRunner.sendGoal(goal_x, goal_y, true_x, true_y, object_name)
                    prediction_error = np.sqrt((goal_x - true_x)**2 + (goal_y - true_y)**2)
                    print(f"📌 Real object position: ({true_x:.2f}, {true_y:.2f})")
                    print("="*80)
                    print(f"📐 Prediction error: {prediction_error:.3f}m")
                else:
                    true_x = true_y = prediction_error = None
                    print("⚠️ Real object position not found in .world file!")
                    break

                print(f"🎯 Navigating to: ({goal_x:.2f}, {goal_y:.2f})") 

        
                for r in nav_results:
                    dx = abs(true_x - r['start_x'])
                    dy = abs(true_y - r['start_y'])

                    euclidean = np.sqrt(dx**2 + dy**2)

                    if r['success']:
                        spl = euclidean / max(r['path_length'], euclidean)
                    else:
                        spl = 0.0

                    retriever.metrics.add_attempt({
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                    'success': r['success'],
                    'path_length': r['path_length'],
                    'optimal_path_length': euclidean, 
                    'final_distance': r['final_distance'] , 
                    'spl': spl,     
                    'start_x': r['start_x'],
                    'start_y': r['start_y'],
                    'final_x': r['final_x'],
                    'final_y': r['final_y'],
                    'object_name': object_name,
                    'true_x': true_x,
                    'true_y': true_y,
                    'taken_time': r['taken_time'],
                    'prediction_error': prediction_error
                })


                torch.cuda.empty_cache()

                print("\n📊 Trial Results:")
                for r in nav_results:
                    print(f"   Trial: {'✅' if r['success'] else '❌'} | "
                        f"Path={r['path_length']:.2f}m | Dist={r['final_distance']:.2f}m")
                    
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

