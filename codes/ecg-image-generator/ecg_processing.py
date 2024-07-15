import os
from gen_ecg_image_from_data import run_single_file

class Args:
  pass

def process_file(file_tuple, args, abs_input_directory, abs_output_directory):
  full_header_file, full_recording_file = file_tuple
  filename = full_recording_file
  header = full_header_file
  
  args_obj = Args()
  args_obj.input_file = os.path.join(abs_input_directory, filename)
  args_obj.header_file = os.path.join(abs_input_directory, header)
  
  folder_struct_list = full_header_file.split('/')[:-1]
  args_obj.output_directory = os.path.join(abs_output_directory, '/'.join(folder_struct_list))
  args_obj.encoding = os.path.split(os.path.splitext(filename)[0])[1]
  args_obj.start_index = -1
  args_obj.config_file = args['config_file']
  
  # Add all other parameters with default values
  for var_name, var_value in args.items():
      if not var_name.startswith('__') and var_name not in ['i', 'o', 'se', 'input_directory', 'output_directory', 'config_file']:
          setattr(args_obj, var_name, var_value)
  
  # Set default values for missing attributes
  default_attributes = {
      'st': False,
      'seed': -1,
      'resolution': 200,
      'random_resolution': False,
      'pad_inches': 0,
      'random_padding': False,
      'remove_lead_names': [],
      'calibration_pulse': 0.5,
      'random_bw': 0.5,
      'random_grid_present': 0.5,
      'print_header': False,
      'random_print_header': 0.5,
      'random_grid_color': False,
      'standard_grid_color': 'red',
      'mask_unplotted_samples': False,
      'store_config': 0,
      'lead_name_bbox': False,
      'full_mode': 'II',
      'lead_bbox': False,
      'num_columns': -1,
      'fully_random': False,
      'hw_text': False,
      'wrinkles': False,
      'augment': False,
      'num_words': 5,
      'deterministic_num_words': False,
      'x_offset': 30,
      'y_offset': 20,
      'deterministic_offset': False,
      'handwriting_size_factor': 1.0,
      'link': '',
      'crease_angle': 45,
      'deterministic_angle': False,
      'num_creases_vertically': 2,
      'deterministic_vertical': False,
      'num_creases_horizontally': 2,
      'deterministic_horizontal': False,
      'noise': 10,
      'deterministic_noise': False,
      'crop': 0,
      'rotate': 0,
      'add_qr_code': False
  }
  
  for attr, default_value in default_attributes.items():
      if not hasattr(args_obj, attr):
          setattr(args_obj, attr, default_value)
  
  return run_single_file(args_obj)