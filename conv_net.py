import numpy as np

def convolution2d(matrix, filter, padding_width, padding_height, stride_width, stride_height):
    input_height, input_width = matrix.shape
    filter_height, filter_width = filter.shape
    output_width = int((input_width + 2 * padding_width - filter_width) / stride_width + 1)
    output_height = int((input_height + 2 * padding_height - filter_height) / stride_height + 1)

    padded_matrix = add_padding(matrix, padding_width, padding_height)

    output = np.zeros((output_width, output_height))
    
    # Apply filter
    for row_index, row in enumerate(matrix):
        for column_index, element in enumerate(row):
            pixel_sum = 0
            for filter_row_index, filter_row in enumerate(filter):
                for filter_column_index, filter_weight in enumerate(filter_row):
                    print(row_index + filter_row_index - ((filter_height - 1) / 2) + padding_height, column_index + filter_column_index - ((filter_width - 1) / 2) + padding_width)
                    pixel_sum += filter_weight * padded_matrix[int(row_index + filter_row_index - ((filter_height - 1) / 2) + padding_height)][int(column_index + filter_column_index - ((filter_width - 1) / 2) + padding_width)]
            output[row_index][column_index] = pixel_sum
    
    print(output)
    return output

def add_padding(matrix, padding_width, padding_height):
    padded_matrix = np.pad(matrix, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')
    return padded_matrix

input_matrix = np.array([
    [3,2,1],
    [1,2,3],
    [2,3,1],
])

filter = np.array([
    [0,0,0],
    [1,1,1],
    [0,0,0]
])

convolution2d(input_matrix, filter, 1, 1, 1, 1)