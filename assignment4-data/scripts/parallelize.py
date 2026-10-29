import concurrent.futures
import os
from tqdm import tqdm
import pathlib
import multiprocessing

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract

from scripts.preprocessing.gopher_quality_filters import is_high_quality
from scripts.preprocessing.quality_classifier import get_valid_record
from scripts.preprocessing.harmful_content_detector import classify_nsfw, classify_hate
from scripts.preprocessing.mask_pii import all_masks

from collections import Counter 

def print_a_few_records(input_path):
    with open(input_path, "rb") as f:
        ai = ArchiveIterator(f, record_types=WarcRecordType.response)
        for _ in range(4):
            record = get_valid_record(ai)
            print(record)

def process_single_wet_file(input_path: str, output_path: str) -> tuple[str, Counter]:
    # TODO: read input path, process the input, and write the output to output_path
    filterReasons = Counter(not_high_quality=0, nsfw=0, hate=0)
    
    with open(input_path, "rb") as f:
        ai = ArchiveIterator(f, record_types=WarcRecordType.response)
        while True:
            try:
                record = get_valid_record(ai)
            except StopIteration:
                break
            # the filters
            if not is_high_quality(record):
                filterReasons["not_high_quality"] += 1
                continue
            if classify_nsfw(record)[0]:
                filterReasons["nsfw"] += 1
                continue
            if classify_hate(record)[0]:
                filterReasons["hate"] += 1
                continue
            
            
    return output_path, filterReasons
# Set up the executor

if False: # Saving this for later
    num_cpus = multiprocessing.cpu_count()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    wet_filepaths = ["a.warc.wet.gz","b.warc.wet.gz","c.warc.wet.gz"]
    output_directory_path = "/path/to/output_directory/"
    futures = []
    for wet_filepath in wet_filepaths:
        # For each warc.wet.gz filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(wet_filepath).name)
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            os.path.join(output_directory_path, wet_filename)
        )
        # Store the futures
        futures.append(future)

    # Iterate over the completed futures as they finish, using a progress bar
    # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
    ):
        output_file = future.result()
        print(f"Output file written: {output_file}")
        
if __name__ == "__main__":
    print_a_few_records("./local-shared-data/")