#!/bin/bash
# Script to fix ImageMagick PDF security policy
# This script modifies the ImageMagick policy.xml file to allow PDF operations
# chmod +x presentation/fix_imagemagick_policy.sh
# presentation/fix_imagemagick_policy.sh

# Find all possible policy.xml locations
POLICY_FILES=( 
    "/etc/ImageMagick-6/policy.xml"
    "/etc/ImageMagick-7/policy.xml"
    "/etc/ImageMagick/policy.xml"
    "/usr/local/etc/ImageMagick-6/policy.xml"
    "/usr/local/etc/ImageMagick-7/policy.xml"
    "/usr/local/etc/ImageMagick/policy.xml"
)

FOUND=0

# Function to check if we have sudo access
check_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        # Already root
        return 0
    elif command -v sudo >/dev/null 2>&1; then
        # Have sudo command
        sudo -v >/dev/null 2>&1
        return $?
    else
        # No sudo
        return 1
    fi
}

# Function to modify the policy file
modify_policy() {
    local file=$1
    echo "Found policy file: $file"
    
    # Check if file contains PDF restriction
    if grep -q "<policy domain=\"coder\" rights=\"none\" pattern=\"PDF\"" "$file"; then
        echo "PDF restriction found in $file"
        
        # Check if we have permission to modify the file
        if check_sudo; then
            # Create backup
            if [ "$(id -u)" -eq 0 ]; then
                cp "$file" "${file}.bak"
            else
                sudo cp "$file" "${file}.bak"
            fi
            echo "Created backup at ${file}.bak"
            
            # Option 1: Modify only PDF rights
            if [ "$(id -u)" -eq 0 ]; then
                sed -i 's/<policy domain="coder" rights="none" pattern="PDF"/<policy domain="coder" rights="read|write" pattern="PDF"/g' "$file"
            else
                sudo sed -i 's/<policy domain="coder" rights="none" pattern="PDF"/<policy domain="coder" rights="read|write" pattern="PDF"/g' "$file"
            fi
            
            echo "Successfully modified policy to allow PDF operations"
            echo "If you need to revert these changes, use the backup at ${file}.bak"
            FOUND=1
            return 0
        else
            echo "ERROR: You need root/sudo privileges to modify $file"
            echo "Try running this script with sudo"
            return 1
        fi
    else
        echo "PDF restriction not found in this file or already modified"
        FOUND=1
        return 0
    fi
}

# Main process
echo "Searching for ImageMagick policy files..."

for file in "${POLICY_FILES[@]}"; do
    if [ -f "$file" ]; then
        modify_policy "$file"
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "ERROR: Could not find any ImageMagick policy files"
    echo "Possible locations checked: ${POLICY_FILES[*]}"
    echo "You may need to manually locate your policy.xml file and modify it"
    exit 1
fi

echo "Done!"
echo "Try running your conversion command again"