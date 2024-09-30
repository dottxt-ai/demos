
import datetime
import os
from pathlib import Path
from textwrap import dedent

# Supported file extensions for human-readable text files
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.rst', '.org', '.json', '.yaml', '.yml',
    '.ini', '.cfg', '.conf', '.log', '.csv', '.tsv'
}

"""
This function takes a list of file contents, then splits the list into pairs
where each pair is (file_content, other_file_samples).
"""
def create_file_pairs(file_metadata):
    for i in range(len(file_metadata)):
        other_file_samples = file_metadata[:i] + file_metadata[i+1:]
        yield file_metadata[i], other_file_samples  

def list_supported_files(directory):
    supported_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                supported_files.append(os.path.join(root, file))
    return supported_files

def get_file_head(file_path, num_chars=250):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(num_chars)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def get_file_creation_time(file_path):
    """
    Get the creation time of a file.
    
    Args:
    file_path (str): The path to the file.
    
    Returns:
    datetime: The creation time of the file as a datetime object.
    """
    try:
        # Get the file's creation time as a timestamp
        creation_time = os.path.getctime(file_path)
        
        # Convert the timestamp to a datetime object
        creation_datetime = datetime.datetime.fromtimestamp(creation_time)
        
        return creation_datetime
    except OSError as e:
        print(f"Error getting creation time for {file_path}: {e}")
        return None
    
def get_file_metadata(file_path, num_chars=250):
    """
    Get metadata for a file including its creation time, head content, and other relevant information.
    
    Args:
    file_path (str): The path to the file.
    num_chars (int): Number of characters to read from the head of the file (default: 250).
    
    Returns:
    dict: A dictionary containing the file's metadata.
    """
    metadata = {}
    path = Path(file_path)
    
    try:
        # File name and extension
        metadata['name'] = path.name
        metadata['extension'] = path.suffix.lower()
        
        # File size
        metadata['size'] = path.stat().st_size
        
        # Creation time
        creation_time = datetime.datetime.fromtimestamp(path.stat().st_ctime)
        metadata['creation_time'] = creation_time
        metadata['creation_date'] = creation_time.strftime('%Y-%m-%d')
        
        # Modification time
        mod_time = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        metadata['modification_time'] = mod_time
        metadata['modification_date'] = mod_time.strftime('%Y-%m-%d')
        
        # File content head
        try:
            with path.open('r', encoding='utf-8') as file:
                metadata['head_content'] = file.read(num_chars)
        except UnicodeDecodeError:
            metadata['head_content'] = "Unable to read file content (non-text file)"
        
        # File type (based on extension)
        metadata['file_type'] = get_file_type(metadata['extension'])
        
    except OSError as e:
        print(f"Error getting metadata for {file_path}: {e}")
        return None
    
    return metadata

def get_file_type(extension):
    """
    Determine the file type based on its extension.
    """
    extension = extension.lower()
    if extension in ['.txt', '.md', '.rst', '.org']:
        return 'document'
    elif extension in ['.json', '.yaml', '.yml']:
        return 'data'
    elif extension in ['.py', '.js', '.java', '.cpp']:
        return 'code'
    elif extension in ['.jpg', '.jpeg', '.png', '.gif']:
        return 'image'
    elif extension in ['.mp3', '.wav', '.ogg']:
        return 'audio'
    elif extension in ['.mp4', '.avi', '.mov']:
        return 'video'
    else:
        return 'other'
