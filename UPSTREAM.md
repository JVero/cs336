# Upstream starter code

Each assignment directory is a plain copy of the official student repo at the commit below, with its `.git` removed. To see what the staff changed since then:

```sh
git clone --depth 1 https://github.com/stanford-cs336/assignment1-basics /tmp/a1-upstream
diff -r --exclude=.git --exclude=.venv --exclude=data /tmp/a1-upstream assignment1-basics
```

| Assignment | Upstream repo | Commit | Upstream date |
|---|---|---|---|
| assignment1-basics | https://github.com/stanford-cs336/assignment1-basics | a158843b20107949f1a8d7df1b05cd33b9166712 | 2026-04-07 |
| assignment2-systems | https://github.com/stanford-cs336/assignment2-systems | ca8bc81a59b70516f7ebb2da4808daade877c736 | 2026-05-01 |
| assignment3-scaling | https://github.com/stanford-cs336/assignment3-scaling | 03e9372992e913061b9e78b5cfcb62ad8a87de35 | 2026-05-08 |
| assignment4-data | https://github.com/stanford-cs336/assignment4-data | 0555bea66369872d912652debf10b115ca0688c8 | 2026-05-07 |
| assignment5-alignment | https://github.com/stanford-cs336/assignment5-alignment | c2734a26308710949fe13226960a1e8cece94b7e | 2026-06-04 |

Copied on 2026-09-01.
