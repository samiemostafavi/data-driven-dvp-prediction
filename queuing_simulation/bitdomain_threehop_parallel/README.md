For large simulations the swap file size must be increased in Ubuntu.
- Turn off all swap processes
	`sudo swapoff -a`
- Resize the swap (to 8 GB for example)
	`sudo dd if=/dev/zero of=/swapfile bs=1G count=8`
if = input file
of = output file
bs = block size
count = multiplier of blocks
- Change permission
	`sudo chmod 600 /swapfile`
- Make the file usable as swap
	`sudo mkswap /swapfile`
- Activate the swap file
	`sudo swapon /swapfile`
- Edit /etc/fstab and add the new swapfile if it isn’t already there
	`/swapfile none swap sw 0 0`
- Check the amount of swap available
	`grep SwapTotal /proc/meminfo`
