import re

def remove_name_time(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        index = line.find("):")
        if index != -1:
            modified_line = line[index + 3:]
        else:
            modified_line = line
        modified_lines.append(modified_line)

    with open(filename, 'w') as file:
        file.writelines(modified_lines)

def remove_leading_whitespace(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    modified_lines = [line.lstrip() for line in lines]

    with open(filename, 'w') as file:
        file.writelines(modified_lines)


def remove_empty_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    non_empty_lines = [line for line in lines if line.strip()]

    with open(filename, 'w') as file:
        file.writelines(non_empty_lines)


# remove_name_time('./datasets/virgin-datasets/calling/calling-training-dataset.txt')
remove_leading_whitespace('./datasets/virgin-datasets/calling/calling-training-dataset.txt')
# remove_empty_lines('./datasets/virgin-datasets/calling/calling-training-dataset.txt')
