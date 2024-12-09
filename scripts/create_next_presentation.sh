#!/bin/bash

# Path to the Reveal.js presentations folder
presentations_dir="/Users/andreasbelavic/bachelor-thesis/presentations"

# Path to the file storing the current week number
counter_file="$presentations_dir/data/week_counter.txt"

# Path to the JavaScript file where the presentations array will be stored
js_file="$presentations_dir/data/presentations.js"

# Check if the counter file exists
if [ ! -f "$counter_file" ]; then
  echo "1" > "$counter_file"  # Initialize the counter file with week 1 if it doesn't exist
fi

# Read the current week number from the counter file
week_number=$(cat "$counter_file")

# Create a new folder for the new week
new_week="week$week_number"
mkdir -p "$presentations_dir/$new_week"

# Add a blank markdown file in the new folder
touch "$presentations_dir/$new_week/presentation.md"

# Add a default template to the markdown file
echo "# Woche $week_number" > "$presentations_dir/$new_week/presentation.md"
cat "$presentations_dir/resources/template.md" >> "$presentations_dir/$new_week/presentation.md"
last_week=$((week_number - 1))
cp "$presentations_dir/week$last_week/gantt.md" "$presentations_dir/week$week_number/"

# Increment the week number and update the counter file
new_week_number=$((week_number + 1))
echo "$new_week_number" > "$counter_file"

# Generate the list of presentations as a JavaScript array
presentations_array="const presentations = ["

# Iterate through the directories and add each to the array
for dir in $(ls -d $presentations_dir/week*/); do
    week=$(basename $dir)
    presentations_array="$presentations_array'$week',"
done

# Remove the trailing comma and close the array
presentations_array="${presentations_array%,}]"

# Write the array to the JavaScript file
echo "$presentations_array;" > "$js_file"

echo "Presentation folder for $new_week created successfully!"
echo "Presentations array updated successfully!"
