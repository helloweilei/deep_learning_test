# Custom Commands for opencode

## /commit
When user types `/commit`, automatically stage all changes and create a commit with a descriptive message.

The commit process should:
1. Check git status to see what files have changed
2. Stage all changes with `git add .`
3. Create a commit with an appropriate message summarizing the changes
4. Show the commit result
5. Push commit to remote with `git push`
6. Show the push result
