import os 
import csv 
from datetime import datetime 
from typing import List, Tuple , Any 



def save_to_csv(list_of_files_info : List[Tuple[Any,...]]):

    current_datetime = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    files_logger_csv_path = f"logger/ingestion_files_{current_datetime}.csv"
    with open(files_logger_csv_path, mode = 'w', newline= '') as file:
        writer = csv.writer(file)
        writer.writerow(["File Extension", "File Name", "Full Path", "Size (bytes)" ])
        writer.writerows(list_of_files_info)

def print_dataset_size_stats(list_of_files_info):
    overall_size = 0

    size_per_file_type = {}

    for file_info in list_of_files_info:
        file_type, filename, curr_file_name_with_path, curr_file_size = file_info 
        overall_size += curr_file_size 
        if file_type in size_per_file_type:
            size_per_file_type[file_type] += curr_file_size
        else:
            size_per_file_type[file_type] = curr_file_size

    
    bytes_per_one_gb = 1024* 1024 * 1024 

    overall_size_in_gbs = overall_size/ bytes_per_one_gb 
    print(f'Overall size : {overall_size_in_gbs} Gbs')
    for file_type, size in size_per_file_type.items():
        size_in_gbs = size / bytes_per_one_gb
        print(f" {file_type} : {size_in_gbs} GBs")




class FileScanner:

    def scan(self, root: str, types: list) -> list:
        """
        Scan the files in the specified root directory and return a dictionary
        containing list of file paths for each file type in the given types list

        Args:
            root(str): root directory to scan
            types(list): list of file types like pdf, csv..
        
        Returns:
            dict: dictionary with keys are file types, values are list of file paths for each file type 
        """

        exclude_prefixes_list = ['~$', '.pdf', '.csv']

        min_file_size = 1024 

        max_file_size = 1024 * 1024 * 1024

        files = {file_type : [] for file_type in types}

        cwd = os.getcwd()

        root_absolute_path = os.path.join(cwd, root)

        list_of_files_info = []

        for dirpath, _, filenames in os.walk(root_absolute_path):
            filtered_files = [f for f in filenames \
                              if not any(f.startswith(exclude_prefix) \
                                         for exclude_prefix in exclude_prefixes_list )]
            
            for filename in filtered_files:
                for file_type in types: 
                    if filename.endswith(file_type):
                        curr_file_with_path = os.path.join(dirpath, filename)
                        curr_file_size = os.path.getsize(curr_file_with_path)
                        if curr_file_size > min_file_size and \
                            curr_file_size < max_file_size:
                            curr_file_with_path = os.path.relpath(curr_file_with_path, cwd)
                            files[file_type].append(curr_file_with_path)
                            list_of_files_info.append((file_type, filename, \
                                                       curr_file_with_path, curr_file_size))
                            
        

        save_to_csv(list_of_files_info)

        print_dataset_size_stats(list_of_files_info)

        return files 

