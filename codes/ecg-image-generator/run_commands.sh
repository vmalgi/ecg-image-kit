#!/bin/bash

# Function to log errors (optional but recommended)
log_error() {
  echo "Error: $1" >> error.log  # Appends the error message to error.log
}

# First Python command
python gen_ecg_images_from_data_batch.py -i "/Users/vinayaka/Desktop/Physionet-24/physionet/records100/05000_p1" -o "/Users/vinayaka/Desktop/Physionet-24/physionet/records100_images/Train/05000_p1" --augment -rot 5 -noise 40 --deterministic_rot --deterministic_noise --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 -se 10 --print_header --add_qr_code --store_config 1 || {
  log_error "Command 1 failed." 
}

# Second Python command
# python gen_ecg_images_from_data_batch.py -i "/Users/vinayaka/Desktop/Physionet-24/physionet/records100/05000_p2" -o "/Users/vinayaka/Desktop/Physionet-24/physionet/records100_images/Train/05000_p2" --augment -rot 5 -noise 40 --deterministic_rot --deterministic_noise --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 -se 10 --print_header --add_qr_code --store_config 1 || {
#   log_error "Command 2 failed." 
# }

# # Third Python command
# python gen_ecg_images_from_data_batch.py -i "/Users/vinayaka/Desktop/Physionet-24/physionet/records100/08000" -o "/Users/vinayaka/Desktop/Physionet-24/physionet/records100_images/Train/08000" --augment -rot 5 -noise 40 --deterministic_rot --deterministic_noise --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 -se 10 --print_header --add_qr_code --store_config 1 || {
#   log_error "Command 3 failed." 
# } 