for a in range(4,100):    
    input_file_path = f'error-{a}.txt'
    with open(input_file_path, 'r') as file:
        content = file.read()

    modified_content = ""
    i = 0
    prevchar = ''
    for char in content:
        if i == 0:
            modified_content += char
        else: 
            if char == '.':
                modified_content = modified_content[:-1]
                modified_content = modified_content + '\n' + prevchar + char
            else:
                modified_content += char
        prevchar = char
        i = i+1

    output_file_path = f'fixed-{a}.txt'
    with open(output_file_path, 'w') as file:
        file.write(modified_content)
