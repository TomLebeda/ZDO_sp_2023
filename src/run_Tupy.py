# Coding: utf-8

import argparse
from find_incision import *
import json

# ****************************************************************************************
# ****                           P A R S I N G   A R G S                              ****
# ****************************************************************************************

# Create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('output_file', help='output file path (JSON format)')
parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode')
parser.add_argument('-sf', '--save_fig', action='store_true', help='enable save figure')
parser.add_argument('input_files', nargs='+', help='input file paths')

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
output_file = args.output_file
verbose = args.verbose
save_fig = args.save_fig
input_files = args.input_files

CHAR_LEN = 30   # only for print to the console

# Check if the output file format is JSON
if not output_file.endswith('.json'):
    print("Error: Output file format must be JSON.")
    exit(1)

# Print the parsed values
print('-' * CHAR_LEN)
print(f"Output file:\t {output_file}")
print(f"Verbose mode:\t {verbose}")
print(f"Save figure:\t {save_fig}")
print(f"Input files:\t {input_files}")
print('-' * CHAR_LEN)

# ****************************************************************************************
# ****                                 A L G O R I T M H                              ****
# ****************************************************************************************

FOLDER = './input_images/'   # folder for input images in args

output_data = list()         # init final output data for all image in args

# loop for all images in args
for file_name in input_files:

    data = init_data()       # init data for actually image
    data['filename'] = file_name
    try:
        data['incision_polyline'] = run_find_incisions(FOLDER + file_name, save_fig, verbose)
        # data['crossing_positions'] = ... prepare to connect
        # data['crossing_angles'] = ... prepare to connect
    except KeyboardInterrupt:
        print('Manually kill.')
    except Exception as e:
        print(f'END: {e}')
        exit(1)

    output_data.append(data)

# save output data
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f'\nOutput data successfully saved: \t{output_file}\n')