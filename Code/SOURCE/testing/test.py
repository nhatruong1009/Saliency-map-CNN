
def write_to_file_three_times(file_name, text):
    with open(file_name, "a") as f:
        f.write(text + "\n")
        
file_name = "weget"
text = "hello"
write_to_file_three_times(file_name, text)
write_to_file_three_times(file_name, text)
write_to_file_three_times(file_name, text)
write_to_file_three_times(file_name, text)
write_to_file_three_times(file_name, text)
