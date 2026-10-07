import os
import zipfile
from pathlib import Path

def create_submission_zip():
    team_name = "MyTeam" # User can rename the zip later
    zip_filename = f"{team_name}_submission.zip"
    base_dir = Path(__file__).resolve().parent
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. output/
        output_dir = base_dir / "output"
        for file in ["matching_results.tsv", "candidate_pairs.tsv"]:
            file_path = output_dir / file
            if file_path.exists():
                zipf.write(file_path, arcname=f"output/{file}")
            else:
                print(f"Warning: {file} not found in output directory.")
                
        # 2. code/business_entity_resolution/
        src_dir = base_dir / "src"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        arcname = f"code/business_entity_resolution/src/{file_path.relative_to(src_dir)}"
                        zipf.write(file_path, arcname=arcname)
                        
        # Include README and requirements in code folder
        for file in ["README.md", "requirements.txt"]:
            file_path = base_dir / file
            if file_path.exists():
                zipf.write(file_path, arcname=f"code/business_entity_resolution/{file}")
                
        # 3. Documentation_template.md at root of zip
        doc_path = base_dir / "Documentation_template.md"
        if doc_path.exists():
            zipf.write(doc_path, arcname="Documentation_template.md")
            
    print(f"Successfully created {zip_filename} adhering to competition rules!")

if __name__ == '__main__':
    create_submission_zip()
