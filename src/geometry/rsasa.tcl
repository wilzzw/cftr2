proc calc_sasa {traj_id r} {
    set sel [atomselect top "protein and resid $r"]
    # Lipids are included; important for residues at the protein lipid interface
    set protein [atomselect top "protein or lipids"]
    set n [molinfo top get numframes]
    set output [open "~/cftr2/results/data/${traj_id}_SASA_$r.dat" w]
    
    # sasa calculation loop
    for {set i 0} {$i < $n} {incr i} {
        molinfo top set frame $i
        set sasa [measure sasa 1.4 $protein -restrict $sel]
        puts "\t \t progress: $i/$n"
        puts $output "$sasa"
    }
    puts "\t \t progress: $n/$n"
    puts "Done."
    close $output
}

