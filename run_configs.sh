#!/bin/bash

# Check if at least one argument (YAML file path) is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 yaml_file1.yaml [yaml_file2.yaml ...]"
    exit 1
fi

# Sort YAML files alphabetically
sorted_files=($(printf "%s\n" "$@" | sort))

echo "Running ${#sorted_files[@]} config files in alphabetical order..."

# Run each YAML file sequentially
for i in "${!sorted_files[@]}"; do
    file="${sorted_files[$i]}"
    echo "[$((i+1))/${#sorted_files[@]}] Running config: $file"
    python main_yaml.py --config "$file"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Execution failed for $file. Skipping to next."
    fi
done

echo "✅ All runs completed (with possible failures)."
