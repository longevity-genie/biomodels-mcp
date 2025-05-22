from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter
import torch
import urllib.request
import gzip
import shutil
import logging
from pathlib import Path
from typing import Dict, Annotated
import typer
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

    logger.info(f"Downloading {url} to {local_path_temp}...")
    try:
        with urllib.request.urlopen(url) as response, open(local_path_temp, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        logger.info(f"Successfully downloaded {local_path_temp}")

        if local_path_temp.suffix == '.gz' and final_target_path.suffix != '.gz':
            logger.info(f"Decompressing {local_path_temp} to {final_target_path}...")
            with gzip.open(local_path_temp, 'rb') as f_in:
                with open(final_target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            local_path_temp.unlink() # Remove the gzipped temporary file
            logger.info(f"Successfully decompressed to {final_target_path}")
        elif local_path_temp != final_target_path: # Covers all other cases where names differ and no decompression occurred
            # Store properties of the source path for accurate logging,
            # as the local_path_temp Path object itself refers to the original path
            # even after the rename operation on the file system.
            source_path_str_for_log = str(local_path_temp)
            source_is_gz = local_path_temp.suffix == '.gz'

            local_path_temp.rename(final_target_path)

            if source_is_gz and final_target_path.suffix == '.gz':
                logger.info(f"Moved/Renamed {source_path_str_for_log} to {final_target_path} (keeping .gz)")
            else:
                logger.info(f"Moved/Renamed {source_path_str_for_log} to {final_target_path}")

    except Exception as e:
        logger.error(f"Error downloading or processing {url} to {final_target_path}: {e}")
        if local_path_temp.exists():
            local_path_temp.unlink() # Clean up temporary file
        return False
    return True

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

    for key, spec in file_specs.items():
        # Derive the download filename from the URL
        parsed_url = urlparse(spec["url"])
        download_filename = Path(parsed_url.path).name
        temp_download_path = data_folder / download_filename
        final_target_path = data_folder / spec["final_filename"]
        output_paths[key + "_path"] = final_target_path

        if final_target_path.exists() and not force_rewrite:
            logger.info(f"File {final_target_path} already exists. Skipping download.")
            continue
        
        logger.info(f"Preparing {key} ({final_target_path})...")
        if not download_file(spec["url"], temp_download_path, final_target_path):
            all_files_ready = False
            logger.warning(f"Failed to obtain {final_target_path}. Subsequent operations may fail.")
            
    if not all_files_ready:
        logger.warning("\nWarning: Not all data files were successfully prepared. The script might fail or use incomplete data.")
    
    return output_paths
# --- End Download Utility ---

@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help=f"Directory to store/find data files. Default: {DEFAULT_DATA_DIR}")] = DEFAULT_DATA_DIR,
    force_download: Annotated[bool, typer.Option("--force-download", "-f", help="Force download of data files even if they already exist.")] = False
):
    """Main execution logic for BioFM example: prepares data, loads model, and runs analysis."""
    
    data_paths = prepare_data_files(data_dir, force_rewrite=force_download)

    critical_files_exist = True
    for key_base in ["gene_annotation", "reference_genome", "vcf"]:
        path_key = key_base + "_path"
        if not data_paths.get(path_key) or not data_paths[path_key].exists():
            logger.error(f"Critical data file for {key_base} ({data_paths.get(path_key, 'Path not generated')}) is missing after preparation. Exiting.")
            critical_files_exist = False
    if not critical_files_exist:
        raise typer.Exit(code=1)

    logger.info("\nLoading BioFM model and tokenizer...")
    model = AnnotatedModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.info("Model and tokenizer loaded.")

    embedder = Embedder(model, tokenizer)

    logger.info("\nSetting up VCF converter...")
    vcf_converter = VCFConverter(
        gene_annotation_path=str(data_paths["gene_annotation_path"]),
        reference_genome_path=str(data_paths["reference_genome_path"])
    )
    logger.info("VCF converter set up.")

    logger.info("\nConverting VCF to annotated dataset...")
    annotated_dataset = vcf_converter.vcf_to_annotated_dataset(
        vcf_path = str(data_paths["vcf_path"]), 
        max_variants=200
    )
    logger.info("VCF converted.")

    logger.info("\nExtracting embeddings...")
    embeddings = embedder.get_dataset_embeddings(annotated_dataset)
    logger.info("Embeddings extracted.")
    logger.info("\n--- Output ---")
    logger.info(f"{embeddings}")
    logger.info("\n----------------")

if __name__ == "__main__":
    app()
