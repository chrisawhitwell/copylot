def _assign_misfit(misfit, allmisfit):
    allfields = list(misfit.keys())
    
    for field in allfields:
        if type(allmisfit[field]) != list:
            dat = allmisfit[field]
            dat_list = [dat]
            dat_list.append(misfit[field])
            allmisfit[field] = dat_list
        else:
            dat = allmisfit[field]
            dat.append(misfit[field])
            allmisfit[field] = dat
    
    return allmisfit 