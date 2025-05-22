from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter
import torch
from pathlib import Path
from typing import Annotated, Dict, Optional, Union
import typer
import eliot
from pycomfort.logging import to_nice_stdout, to_nice_file
from datetime import datetime

# Import from local modules
from biomodels_mcp.config import (
    MODEL_PATH, TOKENIZER_PATH, DEFAULT_DATA_DIR, LOGS_DIR,
    DEFAULT_GENE_ANNOTATION_URL, DEFAULT_REFERENCE_GENOME_URL, DEFAULT_VCF_URL
)
from biomodels_mcp.models import BioFM

app = typer.Typer()

@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help=f"Directory to store/find data files. Default: {DEFAULT_DATA_DIR}")] = DEFAULT_DATA_DIR,
    force_download: Annotated[bool, typer.Option("--force-download", "-f", help="Force download of data files even if they already exist.")] = False,
    logs_dir: Annotated[Path, typer.Option(help=f"Directory to store log files. Default: {LOGS_DIR}")] = LOGS_DIR,
    max_variants: Annotated[int, typer.Option("--max-variants", "-m", help="Maximum number of variants to process from VCF file.")] = 200,
    gene_annotation_url: Annotated[Optional[str], typer.Option("--gene-annotation", help="URL or path to gene annotation file.")] = DEFAULT_GENE_ANNOTATION_URL,
    reference_genome_url: Annotated[Optional[str], typer.Option("--reference-genome", help="URL or path to reference genome file.")] = DEFAULT_REFERENCE_GENOME_URL,
    vcf_url: Annotated[Optional[str], typer.Option("--vcf", help="URL or path to VCF file.")] = DEFAULT_VCF_URL,
    vcf_path: Annotated[Optional[Path], typer.Option("--vcf-path", help="Direct path to VCF file (overrides vcf_url).")] = None
):
    """CLI command that sets up logging and runs the BioFM model."""
    
    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file paths with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"biofm_log_{timestamp}.json"
    rendered_log_file = logs_dir / f"biofm_log_{timestamp}.txt"
    
    # Set up logging
    to_nice_stdout()
    to_nice_file(output_file=log_file, rendered_file=rendered_log_file)
    
    # Create and run the BioFM model with provided options
    biofm = BioFM(
        data_dir=data_dir,
        force_download=force_download,
        logs_dir=logs_dir,
        max_variants=max_variants,
        gene_annotation_url=gene_annotation_url,
        reference_genome_url=reference_genome_url,
        vcf_url=vcf_url
    )
    
    try:
        # Run the model with provided VCF path or default
        biofm.fit(vcf_path)
        annotated_dataset = biofm.dataset
        embeddings = biofm.embeddings
        typer.echo(f"Processing complete. Processed {annotated_dataset.shape} variants and produced {embeddings[0].shape}, {embeddings[1].shape} embeddings.")
    except Exception as e:
        typer.echo(f"Processing failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
