import os
import numpy as np
import os
import pandas as pd
import shutil
import ast
import random
import csv
import multiprocessing as mp
import warnings
from multiprocessing import Pool, cpu_count
#from helper_functions import find_records
from ecg_processing import process_file  # Import the updated process_file function
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

# Force the use of 'forkserver' start method
mp.set_start_method("forkserver", force=True)

# Helper functions
def find_records(folder: str) -> list:
    records = set()
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.hea'):
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                records.add(record)
    return sorted(records)

def find_records_for_image_generation(folder, output_dir):
    header_files = list()
    recording_files = list()

    for root, directories, files in os.walk(folder):
        files = sorted(files)
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.mat':
                record = os.path.relpath(os.path.join(root, file.split('.')[0] + '.mat'), folder)
                hd = os.path.relpath(os.path.join(root, file.split('.')[0] + '.hea'), folder)
                recording_files.append(record)
                header_files.append(hd)
            if extension == '.dat':
                record = os.path.relpath(os.path.join(root, file.split('.')[0] + '.dat'), folder)
                hd = os.path.relpath(os.path.join(root, file.split('.')[0] + '.hea'), folder)
                header_files.append(hd)
                recording_files.append(record)
    
    if recording_files == []:
        raise Exception("The input directory does not have any WFDB compatible ECG files, please re-check the folder!")


    for file in recording_files:
        f, ext = os.path.splitext(file)
        f1 = f.split('/')[:-1]
        f1 = '/'.join(f1)

        if os.path.exists(os.path.join(output_dir, f1)) == False:
            os.makedirs(os.path.join(output_dir, f1))

    return header_files, recording_files

def load_text(filename):
    with open(filename, 'r') as f:
        return f.read()

def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

def get_signal_files(record):
    header_file = record + '.hea'
    header = load_text(header_file)
    return get_signal_files_from_header(header)

def get_signal_files_from_header(string):
    signal_files = []
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i == 0 and not l.startswith('#'):
            num_channels = int(arrs[1])
        elif i <= num_channels and not l.startswith('#'):
            signal_file = arrs[0]
            if signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
    return signal_files

def cast_int_float_unknown(x):
    try:
        return int(x) if float(x).is_integer() else float(x)
    except ValueError:
        return 'Unknown'

def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def get_image_files(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return get_image_files_from_header(header)

def get_image_files_from_header(header):
    image_files = []
    for line in header.splitlines():
        if line.startswith('#Images:'):
            image_files = line.split(': ')[1].split(', ')
            break
    return image_files

def find_files(folder, extensions, remove_extension=False, sort=False):
    selected_files = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension in extensions:
                file = os.path.relpath(os.path.join(root, file), folder)
                if remove_extension:
                    file = os.path.splitext(file)[0]
                selected_files.add(file)
    if sort:
        selected_files = sorted(selected_files)
    return selected_files

# Script 1 functionality
def run_script1(input_folder, output_folder):
    ptbxl_database_file = '/Users/vinayaka/Desktop/Physionet-24/physionet/python-example-2024/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
    ptbxl_mapping_file = '/Users/vinayaka/Desktop/Physionet-24/physionet/python-example-2024/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv'
    sl_database_file = '/Users/vinayaka/Desktop/Physionet-24/physionet/python-example-2024/physionet.org/files/ptb-xl/1.0.3/12sl_statements.csv'
    sl_mapping_file = '/Users/vinayaka/Desktop/Physionet-24/physionet/python-example-2024/physionet.org/files/ptb-xl/1.0.3/12slv23ToSNOMED.csv'

    df_ptbxl_mapping = pd.read_csv(ptbxl_mapping_file, index_col=0)
    subclass_to_superclass = {i: row['diagnostic_class'] for i, row in df_ptbxl_mapping.iterrows() if row['diagnostic'] == 1}

    def assign_superclass(subclasses):
        return list(set(subclass_to_superclass[subclass] for subclass in subclasses if subclass in subclass_to_superclass))

    df_ptbxl_database = pd.read_csv(ptbxl_database_file, index_col='ecg_id')
    df_ptbxl_database.scp_codes = df_ptbxl_database.scp_codes.apply(ast.literal_eval)
    df_ptbxl_database['diagnostic_superclass'] = df_ptbxl_database.scp_codes.apply(assign_superclass)

    df_sl_database = pd.read_csv(sl_database_file, index_col='ecg_id')
    df_sl_mapping = pd.read_csv(sl_mapping_file, index_col='StatementNumber')

    acute_mi_statements = {821, 822, 823, 827, 829, 902, 903, 904, 963, 964, 965, 966, 967, 968}
    acute_mi_classes = {df_sl_mapping.loc[statement]['Acronym'] for statement in acute_mi_statements if statement in df_sl_mapping.index}

    records = find_records(input_folder)

    for record in records:
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df_ptbxl_database.loc[ecg_id]

        date_string, time_string = row['recording_date'].split(' ')
        dd, mm, yyyy = date_string.split('-')[::-1]
        date_string = f'{dd}/{mm}/{yyyy}'

        age = cast_int_float_unknown(row['age'])
        sex = ['Male', 'Female', 'Unknown'][int(row['sex'])]
        height = cast_int_float_unknown(row['height'])
        weight = cast_int_float_unknown(row['weight'])

        scp_codes = [scp_code for scp_code, value in row['scp_codes'].items() if value >= 0]
        superclasses = row['diagnostic_superclass']

        sl_codes = df_sl_database.loc[ecg_id]['statements'] if ecg_id in df_sl_database.index else []

        labels = []
        if 'NORM' in superclasses:
            labels.append('NORM')
        if any(c in sl_codes for c in acute_mi_classes):
            labels.append('Acute MI')
        if 'MI' in superclasses and not any(c in sl_codes for c in acute_mi_classes):
            labels.append('Old MI')      
        if 'STTC' in superclasses:
            labels.append('STTC')
        if 'CD' in superclasses:
            labels.append('CD')
        if 'HYP' in superclasses:
            labels.append('HYP')
        if 'PAC' in scp_codes:
            labels.append('PAC')
        if 'PVC' in scp_codes:
            labels.append('PVC')
        if {'AFIB', 'AFLT'} & set(scp_codes):
            labels.append('AFIB/AFL')
        if {'STACH', 'SVTAC', 'PSVT'} & set(scp_codes):
            labels.append('TACHY')
        if 'SBRAD' in scp_codes:
            labels.append('BRADY') 
        labels = ', '.join(labels)

        input_header_file = os.path.join(input_folder, record + '.hea')
        output_header_file = os.path.join(output_folder, record + '.hea')

        output_path = os.path.join(output_folder, record_path)
        os.makedirs(output_path, exist_ok=True)

        input_header = load_text(input_header_file)
        lines = input_header.split('\n')
        
        record_line = ' '.join(lines[0].strip().split(' ')[:4]) + f' {time_string} {date_string}\n'
        signal_lines = '\n'.join(l.strip() for l in lines[1:] if l.strip() and not l.startswith('#')) + '\n'
        comment_lines = '\n'.join(l.strip() for l in lines[1:] 
            if l.startswith('#') and not any(l.startswith(x) for x in ('# Age:', '# Sex:', '# Height:', '# Weight:', '# Labels:')))
        comment_lines += f'\n# Age: {age}\n# Sex: {sex}\n# Height: {height}\n# Weight: {weight}\n# Labels: {labels}\n'

        output_header = record_line + signal_lines + comment_lines

        save_text(output_header_file, output_header)

        if input_folder != output_folder:
            for signal_file in get_signal_files(os.path.join(input_folder, record)):
                input_signal_file = os.path.join(input_folder, record_path, signal_file)
                output_signal_file = os.path.join(output_folder, record_path, signal_file)
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)

    # print("Script 1: Data preparation completed.")

# Script 2 functionality
# def process_file(file_tuple, args, abs_input_directory, abs_output_directory):
#     # Implement the process_file function here
#     # This is a placeholder, you'll need to add the actual implementation
#     return 1  # Return the number of images generated

def run_script2(args):
  global abs_input_directory, abs_output_directory
  random.seed(args['seed'])

  if not os.path.isabs(args['input_directory']):
      abs_input_directory = os.path.normpath(os.path.join(os.getcwd(), args['input_directory']))
  else:
      abs_input_directory = args['input_directory']

  if not os.path.isabs(args['output_directory']):
      abs_output_directory = os.path.normpath(os.path.join(os.getcwd(), args['output_directory']))
  else:
      abs_output_directory = args['output_directory']
  
  if not os.path.exists(abs_input_directory) or not os.path.isdir(abs_input_directory):
      raise Exception("The input directory does not exist. Please check the input directory path!")

  if not os.path.exists(abs_output_directory):
      os.makedirs(abs_output_directory)

  header_files, recording_files = find_records_for_image_generation(abs_input_directory, abs_output_directory)
  file_tuples = list(zip(header_files, recording_files))

  if args['max_num_images'] != -1:
      file_tuples = file_tuples[:args['max_num_images']]

  if not file_tuples:
      print("No files found to process in the input directory.")
      return

  num_processes = min(cpu_count(), len(file_tuples))

  mp.set_start_method("forkserver", force=True)

  with Pool(num_processes) as pool:
      results = pool.starmap(process_file, [(file_tuple, args, abs_input_directory, abs_output_directory) for file_tuple in file_tuples])

  total_images = sum(results)
  print(f"Script 2: Generated {total_images} ECG images.")

# Script 3 functionality
def run_script3(input_folder, output_folder):
    image_file_types = ['.png', '.jpg', '.jpeg']
    substring_images = '#Images:'

    records = find_records(input_folder)

    image_files = find_files(input_folder, image_file_types)
    record_to_image_files = defaultdict(set)
    for image_file in image_files:
        root, ext = os.path.splitext(image_file)
        record = '-'.join(root.split('-')[:-1])
        basename = os.path.basename(image_file)
        record_to_image_files[record].add(basename)

    for record in records:
        record_path, record_basename = os.path.split(record)
        record_image_files = record_to_image_files[record]

        record_suffixes = [os.path.splitext(image_file)[0].split('-')[-1] for image_file in record_image_files]
        if all(is_number(suffix) for suffix in record_suffixes):
            record_image_files = sorted(record_image_files, key=lambda image_file: float(os.path.splitext(image_file)[0].split('-')[-1]))
        else:
            record_image_files = sorted(record_image_files)
        
        input_header_file = os.path.join(input_folder, record + '.hea')
        output_header_file = os.path.join(output_folder, record + '.hea')

        input_header = load_text(input_header_file)
        output_header = ''
        for l in input_header.split('\n'):
            if not l.startswith(substring_images) and l:
                output_header += l + '\n'

        record_image_string = ', '.join(record_image_files)
        output_header += f'{substring_images} {record_image_string}\n'

        input_path = os.path.join(input_folder, record_path)
        output_path = os.path.join(output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        if os.path.normpath(input_folder) != os.path.normpath(output_folder):
            relative_path = os.path.split(record)[0]

            signal_files = get_signal_files(output_header_file)
            for signal_file in signal_files:
                input_signal_file = os.path.join(input_folder, relative_path, signal_file)
                output_signal_file = os.path.join(output_folder, relative_path, signal_file)
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)

            image_files = get_image_files(output_header_file)
            for image_file in image_files:
                input_image_file = os.path.join(input_folder, relative_path, image_file)
                output_image_file = os.path.join(output_folder, relative_path, image_file)
                if os.path.isfile(input_image_file):
                    shutil.copy2(input_image_file, output_image_file)

    # print("Script 3: Image processing completed."
    
    
def validate_folder_number(value):
    try:
        folder_num = int(value)
        if folder_num < 0 or folder_num > 999999:  # Adjust range as needed
            raise argparse.ArgumentTypeError(f"{value} is not a valid folder number")
        return folder_num
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

def run_pipeline(start_folder, end_folder):
  base_input_path = '/Users/vinayaka/Desktop/Physionet-24/physionet/records100'
  base_output_path = '/Users/vinayaka/Desktop/Physionet-24/physionet/records100_images/Train'

  # Create a list of folder numbers to process
  folders_to_process = list(range(start_folder, end_folder + 1000, 1000))
  
  for folder in tqdm(folders_to_process, desc="Processing folders"):
      # Format the folder number with leading zeros if less than 10000
      folder_str = f"{folder:05d}"
      
      input_folder = os.path.join(base_input_path, folder_str)
      output_folder = os.path.join(base_output_path, folder_str)
      
      if not os.path.exists(input_folder):
          print(f"Input folder {input_folder} does not exist. Skipping.")
          continue
      
      os.makedirs(output_folder, exist_ok=True)
      
      print(f"\nProcessing folder: {folder_str}")
      
      # Run Script 1: Data preparation
      print("Running Script 1: Data preparation")
      run_script1(input_folder, output_folder)
      print("Script 1: Data preparation completed.")
      
      # Run Script 2: Image generation
      print("Running Script 2: Image generation")
      script2_args = {
          'input_directory': output_folder,
          'output_directory': output_folder,
          'x_offset': 30,
          'y_offset': 20,
          'rot': 5,
          'noise': 40,
          'deterministic_rot': True,
          'deterministic_noise': True,
          'n': 4,
          'wrinkles': True,
          'ca': 45,
          'print_header': True,
          'se': 10,
          'random_grid_color': True,
          'add_qr_code': True,
          'seed': -1,
          'max_num_images': -1,
          'config_file': '/Users/vinayaka/Desktop/Physionet-24/physionet/ecg-image-kit/codes/ecg-image-generator/config.yaml'  # Update this path
      }
      run_script2(script2_args)
      print("Script 2: Image generation completed.")
      
      # Run Script 3: Image processing
      print("Running Script 3: Image processing")
      run_script3(output_folder, output_folder)
      print("Script 3: Image processing completed.")
      
      print(f"Completed processing folder: {folder_str}\n")

  print("Pipeline execution completed successfully.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the ECG processing pipeline")
  parser.add_argument("start_folder", type=validate_folder_number, help="Starting folder number (e.g., 3000 or 15000)")
  parser.add_argument("end_folder", type=validate_folder_number, help="Ending folder number (e.g., 3000 or 16000)")
  args = parser.parse_args()

  run_pipeline(args.start_folder, args.end_folder)