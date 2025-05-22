from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter
import torch
import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import Dict, Annotated
import typer
from urllib.parse import urlparse
import eliot
from pycomfort.logging import to_nice_stdout
import sys

# Initialize Eliot for nice console output
eliot.add_destinations(to_nice_stdout)

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"
DEFAULT_DATA_DIR = Path("data")

app = typer.Typer()

# --- Download Utility (adapted from download_data.py) ---
def download_file(url: str, local_path_temp: Path, final_target_path: Path) -> bool:
    """Downloads a file from a URL, handles decompression, and renaming."""
    local_path_temp.parent.mkdir(parents=True, exist_ok=True)
    final_target_path.parent.mkdir(parents=True, exist_ok=True)

    with eliot.start_action(action_type="download_file") as action:
        try:
            action.log(message_type="download_start", url=url, target=str(local_path_temp))
            with urllib.request.urlopen(url) as response, open(local_path_temp, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            action.log(message_type="download_complete", path=str(local_path_temp))

            if local_path_temp.suffix == '.gz' and final_target_path.suffix != '.gz':
                with eliot.start_action(action_type="decompress_file") as decompress_action:
                    decompress_action.log(message_type="decompress_start", source=str(local_path_temp), dest=str(final_target_path))
                    with gzip.open(local_path_temp, 'rb') as f_in:
                        with open(final_target_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    local_path_temp.unlink() # Remove the gzipped temporary file
                    decompress_action.log(message_type="decompress_complete", path=str(final_target_path))
            elif local_path_temp != final_target_path: # Covers all other cases where names differ and no decompression occurred
                with eliot.start_action(action_type="move_rename_file") as move_action:
                    # Store properties of the source path for accurate logging
                    source_path_str_for_log = str(local_path_temp)
                    source_is_gz = local_path_temp.suffix == '.gz'

                    local_path_temp.rename(final_target_path)

                    if source_is_gz and final_target_path.suffix == '.gz':
                        move_action.log(message_type="move_complete", source=source_path_str_for_log, 
                                      dest=str(final_target_path), keep_gz=True)
                    else:
                        move_action.log(message_type="move_complete", source=source_path_str_for_log, 
                                      dest=str(final_target_path))
            return True
        except Exception as e:
            action.log(message_type="download_error", error=str(e), url=url, target=str(final_target_path))
            if local_path_temp.exists():
                local_path_temp.unlink() # Clean up temporary file
            return False

def prepare_data_files(data_folder: Path, force_rewrite: bool = False) -> Dict[str, Path]:
    """
    Ensures all necessary data files are present in data_folder, downloading them if necessary.
    Returns a dictionary mapping file keys to their Path objects.
    """
    data_folder.mkdir(parents=True, exist_ok=True)

    file_specs = {
        "gene_annotation": {
            "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gff3.gz",
            "final_filename": "gencode.v48.annotation.gff3"
        },
        "reference_genome": {
            "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/GRCh38.primary_assembly.genome.fa.gz",
            "final_filename": "GRCh38.primary_assembly.genome.fa"
        },
        "vcf": {
            # Original URL was: "https://raw.githubusercontent.com/m42-health/biofm-eval/main/data/HG01779_b.vcf.gz"
            "url": "https://raw.githubusercontent.com/dna-seq/prs/339f05b6aa68099c31bc77fc5abf28e24bd782a3/data/example.vcf",
            "final_filename": "example.vcf"
        }
    }

    output_paths: Dict[str, Path] = {}
    all_files_ready = True

    with eliot.start_action(action_type="prepare_data_files", folder=str(data_folder)) as action:
        for key, spec in file_specs.items():
            with eliot.start_action(action_type="prepare_file", key=key) as file_action:
                parsed_url = urlparse(spec["url"])
                download_filename = Path(parsed_url.path).name
                temp_download_path = data_folder / download_filename
                final_target_path = data_folder / spec["final_filename"]
                output_paths[key + "_path"] = final_target_path

                if final_target_path.exists() and not force_rewrite:
                    file_action.log(message_type="file_exists", path=str(final_target_path), skipped=True)
                    continue
                
                file_action.log(message_type="file_prepare", file=key, path=str(final_target_path))
                if not download_file(spec["url"], temp_download_path, final_target_path):
                    all_files_ready = False
                    file_action.log(message_type="file_prepare_failed", file=key, path=str(final_target_path))
            
        if not all_files_ready:
            action.log(message_type="prepare_incomplete", warning="Not all data files were successfully prepared")
    
    return output_paths
# --- End Download Utility ---

@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help=f"Directory to store/find data files. Default: {DEFAULT_DATA_DIR}")] = DEFAULT_DATA_DIR,
    force_download: Annotated[bool, typer.Option("--force-download", "-f", help="Force download of data files even if they already exist.")] = False
):
    """Main execution logic for BioFM example: prepares data, loads model, and runs analysis."""
    
    with eliot.start_action(action_type="biomodels_main") as action:
        action.log(message_type="prepare_data_start", data_dir=str(data_dir), force_download=force_download)
        data_paths = prepare_data_files(data_dir, force_rewrite=force_download)

        critical_files_exist = True
        for key_base in ["gene_annotation", "reference_genome", "vcf"]:
            path_key = key_base + "_path"
            if not data_paths.get(path_key) or not data_paths[path_key].exists():
                action.log(message_type="critical_file_missing", file=key_base, path=str(data_paths.get(path_key, "Path not generated")))
                critical_files_exist = False
        if not critical_files_exist:
            raise typer.Exit(code=1)

        with eliot.start_action(action_type="load_model") as model_action:
            model_action.log(message_type="model_loading_start", model_path=MODEL_PATH)
            model = AnnotatedModel.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)
            model_action.log(message_type="model_loading_complete")

        embedder = Embedder(model, tokenizer)

        with eliot.start_action(action_type="setup_vcf_converter") as vcf_action:
            vcf_action.log(message_type="vcf_converter_setup_start")
            vcf_converter = VCFConverter(
                gene_annotation_path=str(data_paths["gene_annotation_path"]),
                reference_genome_path=str(data_paths["reference_genome_path"])
            )
            vcf_action.log(message_type="vcf_converter_setup_complete")

        with eliot.start_action(action_type="convert_vcf") as convert_action:
            convert_action.log(message_type="vcf_conversion_start", vcf_path=str(data_paths["vcf_path"]))
            annotated_dataset = vcf_converter.vcf_to_annotated_dataset(
                vcf_path=str(data_paths["vcf_path"]), 
                max_variants=200
            )
            convert_action.log(message_type="vcf_conversion_complete")

        with eliot.start_action(action_type="extract_embeddings") as embed_action:
            embed_action.log(message_type="embedding_extraction_start")
            embeddings = embedder.get_dataset_embeddings(annotated_dataset)
            embed_action.log(message_type="embedding_extraction_complete")
            action.log(message_type="results", embeddings=str(embeddings))

if __name__ == "__main__":
    app()
