#!/usr/bin/env python

import os
import sys
import logging
import glob
import datetime
from collections import OrderedDict

from solvent_mean import prepare_graphs
#from blob_classification import prepare_classifier

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

import config


MAGIC_COLUMS_CONUT = 56+3*106+1*38
PRINT_FOR_EXCEL = True
TRAIN_SYSTEM = True


def parse_result_line(line):
    result = OrderedDict()
    line_split = line.split(';')
    for part in line_split:
        key, value = part.split(':', 1)
        result[key] = value
    return result


def print_header_for_excel(result, file):
    txt = ';'.join((str(key) for key in result.iterkeys()))
    print >> file, txt


def print_line_for_excel(result, file):
    txt = ';'.join((str(val) for val in result.itervalues()))
    print >> file, txt


def print_line(line, file):
    print >> file, line


def single_file(code):
    data_dir = os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'all')
    output_file_name = os.path.join(data_dir, code + '.csv')
    print_header = True

    for f_name in glob.iglob(os.path.join(data_dir, code + '_results.txt')):
        with open(f_name, 'r') as result_file, open(output_file_name, 'w') as output_file:
            for line in result_file.read().splitlines():
                line = line.strip()
                if len(line) > 0:
                    result = parse_result_line(line)

                    if len(result) == MAGIC_COLUMS_CONUT:
                        if print_header:
                            print_header_for_excel(result, output_file)
                            print_header = False
                        if 'res_name' in result:
                            print_line_for_excel(result, output_file)

    return output_file_name


def postrun():
    start = datetime.datetime.now()
    logger = logging.getLogger('postrun')
    stderr_handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(stderr_handler)
    logger.info("POSTRUN START")
    logger.info(start)
    summary_files = dict()

    data_dir = os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'all')

    for f_name in glob.iglob(os.path.join(data_dir, '*results.txt')):
        result_file = open(f_name, 'r')
        for line in result_file.read().splitlines():
            line = line.strip()
            if len(line) > 0:
                result = parse_result_line(line)

                if len(result) == MAGIC_COLUMS_CONUT:
                    if 'all_data' not in summary_files:
                        summary_files['all_data'] = open(os.path.join(data_dir, 'all_summary.txt'), 'w')
                        if PRINT_FOR_EXCEL is True:
                            print_header_for_excel(result, summary_files['all_data'])

                    if 'res_name' in result:
                        result_type = result['res_name']
                        if result_type not in summary_files:
                            summary_file_name = os.path.join(data_dir, '%s_summary.txt' % result_type)
                            summary_files[result_type] = open(summary_file_name, 'w')

                            if PRINT_FOR_EXCEL is True:
                                print_header_for_excel(result, summary_files[result_type])
                        else:
                            summary_file_name = os.path.join(data_dir, '%s_summary.txt' % result_type)
                            summary_files[result_type] = open(summary_file_name, 'a')

                        summary_file = summary_files[result_type]
                        if PRINT_FOR_EXCEL is True:
                            print_line_for_excel(result, summary_file)
                        else:
                            print_line(line, summary_file)
                        summary_file.close()

                        summary_file = summary_files['all_data']
                        if PRINT_FOR_EXCEL is True:
                            print_line_for_excel(result, summary_file)
                        else:
                            print_line(line, summary_file)
                else:
                    print len(result), result['res_name'] if 'res_name' in result else ''

    data_dir = os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'all')
    for f_name in glob.iglob(os.path.join(data_dir, '*global.txt')):
        result_file = open(f_name, 'r')
        for line in result_file.read().splitlines():
            line = line.strip()
            if len(line) > 0:
                result = parse_result_line(line)

                if 'global_data' not in summary_files:
                    summary_file_name = os.path.join(data_dir, 'global_data.txt')
                    summary_files['global_data'] = open(summary_file_name, 'w')
                    if PRINT_FOR_EXCEL is True:
                        print_header_for_excel(result, summary_files['global_data'])
                summary_file = summary_files['global_data']
                if PRINT_FOR_EXCEL is True:
                    print_line_for_excel(result, summary_file)
                else:
                    print_line(line, summary_file)
        result_file.close()

    # close
    for summary_file in summary_files.itervalues():
        summary_file.close()

    logger.info("POSTRUN STOP")

    logger.info("POSTRUN FIX")
    from ligand_dict import get_ligand_atoms_dict
    src_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dict', 'ligands.txt')
    ligands = get_ligand_atoms_dict(src_file)
    non_h_electron_sum = {}
    for res_name in ligands.iterkeys():
        form = ligands[res_name][0]
        non_h_electron_sum[res_name] = form.get_non_h_electron_count() - form.charge
    if non_h_electron_sum['SO4'] == 48:
        import pandas as pd
        data_dir = os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'all')
        all_summary_path = os.path.join(data_dir, 'all_summary.txt')
        result_data = pd.read_csv(all_summary_path, sep=';', header=0, na_values=['n/a', 'nan'])
        result_data['non_h_electron_sum_fix'] = result_data['res_name'].map(non_h_electron_sum)
        result_data.to_csv(all_summary_path.replace(".", "_fix."), sep=';', index=False, na_rep='nan')

    logger.info("GENERATE GRAPHS")
    # generate graphs
    graphs_dir = os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'graphs')
    prepare_graphs(data_dir, graphs_dir)

    graphs_dir = os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'classifier')
    grouping_path = os.path.join(config.pdb_out_data_dir, 'grouping.txt')
    input_pdb_data_path = os.path.join(data_dir, 'all_summary.txt')
    #if TRAIN_SYSTEM is True:
    #    prepare_classifier(input_pdb_data_path, graphs_dir, grouping_path)
    print('Postrun time: %s s' % (datetime.datetime.now()-start).total_seconds())

if __name__ == '__main__':
    postrun()
