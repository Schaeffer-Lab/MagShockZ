#!/bin/bash
# Git history cleanup script

# Make script exit on any error
set -e

echo "==== Git History Cleanup Tool ===="
echo "WARNING: This will permanently modify your Git history!"
echo "Anyone who has cloned this repository will need to re-clone after this operation."
echo ""
echo "Make sure you have a backup before proceeding!"
echo ""
read -p "Are you sure you want to continue? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "Operation canceled."
    exit 0
fi

# Method 1: Keep only recent commits
read -p "Choose option: (1) Keep only recent commits (2) Remove most recent commit: " option

if [[ "$option" == "2" ]]; then
    echo "Removing most recent commit..."
    read -p "Keep changes in working directory? (y/n): " keep_changes
    
    if [[ "$keep_changes" == "y" ]]; then
        # Soft reset keeps the changes in your working directory
        git reset --soft HEAD~1
        echo "Most recent commit removed. Changes are kept in your working directory."
        echo "You can make modifications and commit again."
    else
        # Hard reset completely removes the commit and its changes
        git reset --hard HEAD~1
        echo "Most recent commit and its changes completely removed."
    fi
    
    echo "Done! You may need to force-push to update remote: git push -f origin $(git branch --show-current)"
    exit 0
fi

read -p "Enter number of recent commits to keep (default: 10): " num_commits
num_commits=${num_commits:-10}

echo "Creating a new branch with only the last $num_commits commits..."
git checkout --orphan temp_branch $(git rev-parse HEAD)
git reset --soft HEAD~$num_commits
git commit -m "Reset history, keeping last $num_commits commits"

# Optionally, verify repository size
echo "Verifying repository size..."
du -sh .git

read -p "Replace main branch with this new history? (y/n): " replace_main
if [[ "$replace_main" == "y" ]]; then
    current_branch=$(git branch --show-current)
    echo "Replacing $current_branch with temp_branch..."
    git branch -D $current_branch
    git branch -m $current_branch
    echo "Done! You'll need to force-push to update remote: git push -f origin $current_branch"
else
    echo "New branch 'temp_branch' created but not applied."
    echo "You can check the result and manually merge if satisfied."
fi

echo ""
echo "==== Additional Options ===="
echo "1. Run garbage collection to free space: git gc --aggressive"
echo "2. Force-push to update remote: git push -f origin $(git branch --show-current)"
echo "3. Notify collaborators to re-clone the repository"
