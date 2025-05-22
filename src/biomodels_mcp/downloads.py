import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
import eliot
from biomodels_mcp.config import (
    DEFAULT_GENE_ANNOTATION_URL,
    DEFAULT_REFERENCE_GENOME_URL,
    DEFAULT_VCF_URL
)

def get_final_filename(url: str) -> str:
    """
    Extracts a sensible filename from a URL, removing .gz extension if appropriate
    
    Args:
        url: URL to extract filename from
        
    Returns:
        A clean filename without .gz extension if it's a compressed file
    """
    parsed_url = urlparse(url)
    temp_filename = Path(parsed_url.path).name
    
    # If it's a gzipped file, return name without .gz
    if temp_filename.endswith('.gz'):
        return temp_filename[:-3]  # Remove .gz extension
    
    return temp_filename

def download_file(url: str, data_folder: Path, force_rewrite: bool = False) -> Optional[Path]:
    """
    Downloads a file from a URL, handles decompression, and file management.
    
    Args:
        url: URL to download from
        data_folder: Folder to download to
        force_rewrite: Whether to force rewriting existing files
        
    Returns:
        Path to the downloaded file if successful, None otherwise
    """
    # Set up paths
    parsed_url = urlparse(url)
    temp_filename = Path(parsed_url.path).name
    final_filename = get_final_filename(url)
    
    temp_path = data_folder / temp_filename
    final_path = data_folder / final_filename
    
    # Create directories
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    with eliot.start_action(action_type="file_download", url=url) as action:
        # Check if file already exists
        if final_path.exists() and not force_rewrite:
            action.log(message_type="file_exists", path=str(final_path))
            return final_path
        
        try:
            # Download the file
            action.log(message_type="download_start", url=url, target=str(temp_path))
            with urllib.request.urlopen(url) as response, open(temp_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            action.log(message_type="download_complete", path=str(temp_path))

            # Handle decompression or renaming
            if temp_path.suffix == '.gz' and final_path.suffix != '.gz':
                # Decompress gzipped file
                action.log(message_type="decompress_start", source=str(temp_path), dest=str(final_path))
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(final_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                temp_path.unlink() # Remove the gzipped temporary file
                action.log(message_type="decompress_complete", path=str(final_path))
            elif temp_path != final_path: # Covers all other cases where names differ and no decompression occurred
                # Store properties of the source path for accurate logging
                source_path_str_for_log = str(temp_path)
                source_is_gz = temp_path.suffix == '.gz'

                temp_path.rename(final_path)

                if source_is_gz and final_path.suffix == '.gz':
                    action.log(message_type="move_complete", source=source_path_str_for_log, 
                              dest=str(final_path), keep_gz=True)
                else:
                    action.log(message_type="move_complete", source=source_path_str_for_log, 
                              dest=str(final_path))
            return final_path
        except Exception as e:
            action.log(message_type="download_error", error=str(e), url=url, target=str(final_path))
            if temp_path.exists():
                temp_path.unlink() # Clean up temporary file
            return None

def prepare_data_files(data_folder: Path, force_rewrite: bool = False) -> Dict[str, Path]:
    """
    Ensures all necessary data files are present in data_folder, downloading them if necessary.
    Returns a dictionary mapping file keys to their Path objects.
    """
    data_folder.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}

    with eliot.start_action(action_type="prepare_data_files", folder=str(data_folder)) as action:
        # Download gene annotation
        gene_annotation_path = download_file(DEFAULT_GENE_ANNOTATION_URL, data_folder, force_rewrite)
        if gene_annotation_path:
            output_paths["gene_annotation_path"] = gene_annotation_path
            
        # Download reference genome
        reference_genome_path = download_file(DEFAULT_REFERENCE_GENOME_URL, data_folder, force_rewrite)
        if reference_genome_path:
            output_paths["reference_genome_path"] = reference_genome_path
            
        # Download VCF example
        vcf_path = download_file(DEFAULT_VCF_URL, data_folder, force_rewrite)
        if vcf_path:
            output_paths["vcf_path"] = vcf_path
            
        if len(output_paths) < 3:
            action.log(message_type="prepare_incomplete", warning="Not all data files were successfully prepared")
    
    return output_paths 