import numpy as np
np.set_printoptions(precision=4)
# def get_probabilities(segment_durations, probabilities, size, options, start_times, time, starting_level, max_level):
#     time = np.array(range(start_time, end_time))
#     zeros = np.zeros(end_time - start_time)
#     ones = np.ones(end_time - start_time)
#     probability = zeros
#     births = []
#     deaths = []
#     for i in range(len(segment_durations)):
#         probability += size[i]*saturate_probabilities(probabilities[i], start_times[i], segment_durations[i], time)
#         # for j in range(options.shape(0)):

def getPlot(base, base_frac_initial, base_frac, base_spawn_options, start_times, t, level, max_level):
    # print(t.shape[0])
    # print('base',base)
    # print('base frac initial',base_frac_initial)
    # print('base frac',base_frac)
    # print('base spawn options',base_spawn_options)
    # print('start times',start_times)
    z = np.zeros(t.shape[0])
    o = np.ones(t.shape[0])
    p = z
    add = np.array([])
    # sub = np.array([])
    sub = []
    # options = np.array([])
    # offset = np.array([])
    options = []
    offset = []
    eq = 0;
    earliest_death = 0;
    for i in range(base.shape[0]):
        # print('test')
        # print(i, start_times[i], base)
        p += base_frac[i]*saturate_probabilities(
            base_frac_initial[i], start_times[i],
            base[i], t);
        for j in range(len(base_spawn_options[base[i]])):
            p -= base_frac[i]*(1./len(base_spawn_options[base[i]]))*saturate_probabilities(
                base_frac_initial[i], start_times[i]+base_spawn_options[base[i]][j],
                base[i],t)
    #     add = np.append(add, np.array([start_times(i); base(i);base_frac(i);sym(base_frac_initial(i))]))
    #     add = [add [start_times(i); base(i);base_frac(i);sym(base_frac_initial(i))]];
    #     num_options = size(base_spawn_options(i,:),2);
    #     for j in range(num_options):
            sub.append([base_spawn_options[base[i]][j] + start_times[i],
                        base[i],base_frac[i]*1./len(base_spawn_options[base[i]]),
                        base_frac_initial[i]])
            # print(base_spawn_options[base[i]][j])
            # if base_spawn_options[base[i]][j] == 3:
            #     options.append([2,3])
            #     offset.append([base_spawn_options[base[i]][j]+start_times[i], base_spawn_options[base[i]][j]+start_times[i]])
            # if base_spawn_options[base[i]][j] == 2:
            #     options.append([3,3])
            #     offset.append([base_spawn_options[base[i]][j]+start_times[i], base_spawn_options[base[i]][j]+start_times[i]])

    #         if base_spawn_options(i,j) == 30
    #             options = [options; 10,30];
    #             offset = [offset; base_spawn_options(i,j)+start_times(i),base_spawn_options(i,j)+start_times(i)];
    #         end
    #         if base_spawn_options(i,j) == 10
    #             options = [options; 30, 30];
    #             offset = [offset; base_spawn_options(i,j)+start_times(i),base_spawn_options(i,j)+start_times(i)];
    #         end
    #         earliest_death = min(earliest_death, start_times(i)+base_spawn_options(i,j))
    #         p = p - base_frac(i)*(1/num_options)*sat(base_frac_initial(i), start_times(i)+base_spawn_options(i,j), base(i),t);
    #     end
    # end
    p = 2**level * p;

    sub = np.array(sub)

    if min(sub[:,0]) <= t[-1]:
        expected_data = getPlot(sub[:,1], sub[:,3], sub[:,2], base_spawn_options, sub[:,0], t, level + 1, max_level)
        # p = [p; expected_data];
        expected_data.insert(0, p)
        return expected_data
    return [p]

def saturate_probabilities(starting_probability, time_offset, segment_duration, time):
    probability = (time - time_offset)/segment_duration
    probability[np.where(probability<0)] = 0.
    probability[np.where(probability>1)] = 1.
    probability *= starting_probability
    return probability

t = np.array(range(1,130))
# saturate_probabilities(1./7, 4, 3, time)
data = getPlot(
    np.array([10.,30.]),#base
    np.array([1./7,6./7]),#base frac initial
    np.array([1.,1.]),#base frac
    {10:[30,30],30:[30,10]},
    # np.array([[30.,30.],[30.,10.]]),#base spawn options
    np.array([0.,0.])#start times
    ,t, 1, 3)
print(np.array(data))
print((data[1][10]-data[1][0])/(t[10]-t[0]))
print((data[1][30]-data[1][10])/(t[30]-t[10]))
print((data[1][40]-data[1][30])/(t[40]-t[30]))
print((data[1][50]-data[1][40])/(t[50]-t[40]))
print((data[1][60]-data[1][50])/(t[60]-t[50]))
print((data[1][70]-data[1][60])/(t[70]-t[60]))
print((data[1][80]-data[1][70])/(t[80]-t[70]))
print((data[1][90]-data[1][80])/(t[90]-t[80]))
