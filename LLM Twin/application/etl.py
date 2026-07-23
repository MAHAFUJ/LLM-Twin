import json
import os
from datetime import datetime

class PersonalDataETL:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_mock_data(self) -> list[dict]:
        """Simulates extracting data from sources."""
        return [
            {
                "source": "linkedin",
                "content": "Just published a new data-driven web app using React and Firebase! Exploring how autonomous electric vehicles can mitigate climate change.",
                "date": "2025-12-10"
            },
            {
                "source": "blog",
                "content": "Why I still prefer physical books for research. The sensory experience of highlighting and sketching on tangible pages improves my comprehension of complex AI architectures.",
                "date": "2026-03-15"
            }
        ]

    def load_to_warehouse(self, data: list[dict]) -> str:
        """Saves extracted data to a local JSON file."""
        file_path = os.path.join(self.output_dir, f"ingestion_{datetime.now().strftime('%Y%m%d')}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Ingested {len(data)} records to {file_path}")
        return file_path

if __name__ == "__main__":
    etl = PersonalDataETL()
    raw_data = etl.extract_mock_data()
    etl.load_to_warehouse(raw_data)