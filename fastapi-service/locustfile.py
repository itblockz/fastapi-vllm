from locust import HttpUser, task, between
import csv
import random

class QuestionUser(HttpUser):
    def on_start(self):
        """Load questions from CSV file when user starts"""
        self.questions = []
        try:
            #path = '/home/siamai/fastapi-vllm/fastapi-service/exanple_question.csv'
            path = '/mnt/data/SCB-dataset/test.csv'
            with open(path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('query'):  # Check if query column exists and has value
                        self.questions.append(row['query'])
            
            if not self.questions:
                raise ValueError("No questions found in CSV file")
                
        except FileNotFoundError:
            print(f"Error: {path} file not found")
            self.questions = ["What is the capital of France?"]  # Fallback question
        except Exception as e:
            print(f"Error reading CSV: {e}")
            self.questions = ["What is the capital of France?"]  # Fallback question

    @task
    def evaluate_question(self):
        """Send random question from CSV to /eval endpoint"""
        question = random.choice(self.questions)
        
        payload = {
            "question": question
        }
        
        with self.client.post("/eval", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    # Validate response structure
                    if "answer" in json_response and "raw_output" in json_response:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
