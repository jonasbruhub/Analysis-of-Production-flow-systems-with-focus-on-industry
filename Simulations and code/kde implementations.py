""" kde.py """
import matplotlib.pyplot as plt

def kde(data, n=2**14, MIN=None, MAX=None, plot=False, label = []):
#==============================================================================
# Reliable and extremely fast kernel density estimator for one-dimensional data;
#        Gaussian kernel is assumed and the bandwidth is chosen automatically;
#        Unlike many other implementations, this one is immune to problems
#        caused by multimodal densities with widely separated modes (see example). The
#        estimation does not deteriorate for multimodal densities, because we never assume
#        a parametric model for the data.
# INPUTS:
#     data    - a vector of data from which the density estimate is constructed;
#          n  - the number of mesh points used in the uniform discretization of the
#               interval [MIN, MAX]; n has to be a power of two; if n is not a power of two, then
#               n is rounded up to the next power of two, i.e., n is set to n=2^ceil(log2(n));
#               the default value of n is n=2^12;
#   MIN, MAX  - defines the interval [MIN,MAX] on which the density estimate is constructed;
#               the default values of MIN and MAX are:
#               MIN=min(data)-Range/10 and MAX=max(data)+Range/10, where Range=max(data)-min(data);
#    plot     -  (True/False) whether or not to produce a plot 
#    label    -  (string) label to apply to kde plot 
    
# OUTPUTS:
#   bandwidth - the optimal bandwidth (Gaussian kernel assumed);
#     density - column vector of length 'n' with the values of the density
#               estimate at the grid points;
#     xmesh   - the grid over which the density estimate is computed;
#             - If no output is requested, then the code automatically plots a graph of
#               the density estimate.
#        cdf  - column vector of length 'n' with the values of the cdf
#  Reference: 
# Kernel density estimation via diffusion
# Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
# Annals of Statistics, Volume 38, Number 5, pages 2916-2957. 

#
#  Example:
#              kde(data,2**14,min(data)-5,max(data)+5);
#
#  Notes:   If you have a more reliable and accurate one-dimensional kernel density
#           estimation software, please email me at botev@maths.uq.edu.au

#==============================================================================
    import numpy as np
    from scipy import optimize
	
	# Auxiliary Functions 
    #==========================================================================  
    def fixed_point(t,N,I,a2):
        # This implements the function t-zeta*gamma^[l](t)
        l=7
        f=2*np.pi**(2*l) * sum(I**l*a2*np.exp(-I*np.pi**2*t))
        
        for s in range(l-1,1,-1):
            K0    = np.prod(range(1,2*s,2))/np.sqrt(2*np.pi)
            const = (1+(1/2)**(s+1/2))/3
            time  = (2*const*K0/N/f)**(2/(3+2*s))
            f     = 2*np.pi**(2*s)*sum(I**s*a2*np.exp(-I*np.pi**2*time)) 
        
        return t-(2*N*np.sqrt(np.pi)*f)**(-2/5)
    #==========================================================================  
    def dct1d(data):
        # computes the discrete cosine transform of the column vector data
        nrows = len(data)
        # Compute weights to multiply DFT coefficients
        weight = 2*(np.exp(-1j*np.array(range(1,nrows))*np.pi/(2*nrows)))
        weight = np.append(1,weight)
        # Re-order the elements of the columns of x
        data = np.append(data[::2],data[:0:-2])
        # Multiply FFT by weights:
        data = (weight*np.fft.fft(data)).real
		
        return data
    #==========================================================================  
    def idct1d(data):
		# computes the discrete cosine transform of the column vector data
        nrows = len(data) 
		# Compute weights to multiply DFT coefficients
        weights = nrows*np.exp(1j*(np.arange(nrows))*np.pi/(2*nrows))
		# Multiply FFT by weights:
        data = np.real(np.fft.ifft(weights * data))
        # Re-order the elements of the columns of x
        output = np.arange(nrows, dtype = 'd')
        output[::2] = data[0:int(nrows/2)]
        output[1::2] = data[:int(nrows/2)-1:-1]
		
		#   Reference:
		#     A. K. Jain, "Fundamentals of Digital Image
		#     Processing", pp. 150-153.
		
        return output
		
   # Main Function
   #===========================================================================  
    data = np.array(data)   #Make data a numpy array 
    
    n=int(2**np.ceil(np.log2(n))) #round up n to the next power of 2;
    
	#define the default  interval [MIN,MAX]
    
    if MAX == None or MIN == None:
        minimum = min(data)
        maximum = max(data)
        Range   = maximum - minimum
    
        if MAX == None:
            MAX=maximum+Range/10
        
        if MIN == None:
            MIN=minimum-Range/10
       

	# set up the grid over which the density estimate is computed;
    R=MAX-MIN; dx=R/(n-1)
    xmesh=np.arange(MIN,MAX+dx,dx, dtype = 'd')
    bins = np.append(xmesh, xmesh[-1])
    N=len(np.unique(data))
	# bin the data uniformly using the grid defined above;
    initial_data= np.histogram(data, bins = bins)[0]/N
    initial_data=initial_data/sum(initial_data)

	# discrete cosine transform of initial data
    a=dct1d(initial_data)

	# now compute the optimal bandwidth^2 using the referenced method
    I=np.arange(1,n,dtype = "d")**2; a2=(a[1:]/2)**2

	# solve the equation t=zeta*gamma^[5](t)
    t_star = optimize.root(lambda t: fixed_point(t,N,I,a2), 0.05)
    if t_star.success == False:
        t_star = 0.28*N**(-2/5)
    else: 
        t_star = t_star.x
	# smooth the discrete cosine transform of initial data using t_star
    a_t=a*np.exp(-np.arange(0,n, dtype = "d")**2*np.pi**2*t_star/2)
    
	# now apply the inverse discrete cosine transform
    density=idct1d(a_t)/R
	# take the rescaling of the data into account
    bandwidth=np.sqrt(t_star)*R
    
    # for cdf estimation
    f=2*np.pi**2*sum(I*a2*np.exp(-I*np.pi**2*t_star))
    t_cdf=(np.sqrt(np.pi)*f*N)**(-2/3)
	# now get values of cdf on grid points using IDCT and cumsum function
    a_cdf=a*np.exp(-np.arange(0,n,dtype="d")**2*np.pi**2*t_cdf/2)
    cdf=np.cumsum(idct1d(a_cdf))*(dx/R)
	#take the rescaling into account if the bandwidth value is required
    bandwidth_cdf=np.sqrt(t_cdf)*R

    if plot==True:
        if label: 
            plt.plot(xmesh, density, label = label)
            plt.legend()
        else: 
             plt.plot(xmesh, density)
        plt.ylim(bottom=0)  
      

    return [bandwidth,density,xmesh,cdf]
	#==========================================================================  




# %#######################################
#     function binned_data=ndhist(data,M)
#     % this function computes the histogram
#     % of an n-dimensional data set;
#     % 'data' is nrows by n columns
#     % M is the number of bins used in each dimension
#     % so that 'binned_data' is a hypercube with
#     % size length equal to M;
#     [nrows,ncols]=size(data);
#     bins=zeros(nrows,ncols);
#     for i=1:ncols
#         [dum,bins(:,i)] = histc(data(:,i),[0:1/M:1],1);
#         bins(:,i) = min(bins(:,i),M);
#     end
#     % Combine the  vectors of 1D bin counts into a grid of nD bin
#     % counts.
#     binned_data = accumarray(bins(all(bins>0,2),:),1/nrows,M(ones(1,ncols)));
#     end

%#######################################
def ndhist(data,M):
    # % this function computes the histogram
    # % of an n-dimensional data set;
    # % 'data' is nrows by n columns
    # % M is the number of bins used in each dimension
    # % so that 'binned_data' is a hypercube with
    # % size length equal to M;
    nrows,ncols=data.shape
    bins=np.zeros(data.shape)
    for i in range(ncols):
        dum, bins[:,i] = histc(data(:,i),[0:1/M:1],1)
        bins[:,i] = min(bins(:,i),M)
    
    # % Combine the  vectors of 1D bin counts into a grid of nD bin
    # % counts.
    binned_data = accumarray(bins(all(bins>0,2),:),1/nrows,M(ones(1,ncols)));
    
    return binned_data


# %#######################################
#     function data=dct2d(data)
#     % computes the 2 dimensional discrete cosine transform of data
#     % data is an nd cube
#     [nrows,ncols]= size(data);
#     if nrows~=ncols
#         error('data is not a square array!')
#     end
#     % Compute weights to multiply DFT coefficients
#     w = [1;2*(exp(-i*(1:nrows-1)*pi/(2*nrows))).'];
#     weight=w(:,ones(1,ncols));
#     data=dct1d(dct1d(data)')';
#         function transform1d=dct1d(x)

#             % Re-order the elements of the columns of x
#             x = [ x(1:2:end,:); x(end:-2:2,:) ];

#             % Multiply FFT by weights:
#             transform1d = real(weight.* fft(x));
#         end
#     end

    
def dct2d(data):
    # % computes the 2 dimensional discrete cosine transform of data
    # % data is an nd cube
    nrows, ncols= data.shape
    if nrows != ncols:
        print('data is not a square array!')
    
    # % Compute weights to multiply DFT coefficients
    w = np.array([1,2*(np.exp(-1j * np.arange(1,nrows,1)*np.pi/(2*nrows)))]);
    weight=w[:,np.ones((1,ncols))];

    def dct1d(x):

        
        x = np.vstack([ x[1:2:-1,:], x[-1:-2:2,:] ] );

        
        transform1d = np.real(weight * scipy.fft(x));
        
        return transform1d

    data=dct1d(dct1d(data).T).T;
        
    
    return data

def kde2d(data,n = 2**8, MIN_XY = None, MAX_XY = None):
    # fast and accurate state-of-the-art
    # bivariate kernel density estimator
    # with diagonal bandwidth matrix.
    # The kernel is assumed to be Gaussian.
    # The two bandwidth parameters are
    # chosen optimally without ever
    # using/assuming a parametric model for the data or any "rules of thumb".
    # Unlike many other procedures, this one
    # is immune to accuracy failures in the estimation of
    # multimodal densities with widely separated modes (see examples).
    # INPUTS: data - an N by 2 array with continuous data
    #            n - size of the n by n grid over which the density is computed
    #                n has to be a power of 2, otherwise n=2^ceil(log2(n));
    #                the default value is 2^8;
    # MIN_XY,MAX_XY- limits of the bounding box over which the density is computed;
    #                the format is:
    #                MIN_XY=[lower_Xlim,lower_Ylim]
    #                MAX_XY=[upper_Xlim,upper_Ylim].
    #                The dafault limits are computed as:
    #                MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
    #                MAX_XY=MAX+Range/4; MIN_XY=MIN-Range/4;
    # OUTPUT: bandwidth - a row vector with the two optimal
    #                     bandwidths for a bivaroate Gaussian kernel;
    #                     the format is:
    #                     bandwidth=[bandwidth_X, bandwidth_Y];
    #          density  - an n by n matrix containing the density values over the n by n grid;
    #                     density is not computed unless the function is asked for such an output;
    #              X,Y  - the meshgrid over which the variable "density" has been computed;
    #                     the intended usage is as follows:
    #                     surf(X,Y,density)
    # Example (simple Gaussian mixture)
    # clear all
    #   % generate a Gaussian mixture with distant modes
    #   data=[randn(500,2);
    #       randn(500,1)+3.5, randn(500,1);];
    #   % call the routine
    #     [bandwidth,density,X,Y]=kde2d(data);
    #   % plot the data and the density estimate
    #     contour3(X,Y,density,50), hold on
    #     plot(data(:,1),data(:,2),'r.','MarkerSize',5)
    #
    # Example (Gaussian mixture with distant modes):
    #
    # clear all
    #  % generate a Gaussian mixture with distant modes
    #  data=[randn(100,1), randn(100,1)/4;
    #      randn(100,1)+18, randn(100,1);
    #      randn(100,1)+15, randn(100,1)/2-18;];
    #  % call the routine
    #    [bandwidth,density,X,Y]=kde2d(data);
    #  % plot the data and the density estimate
    #  surf(X,Y,density,'LineStyle','none'), view([0,60])
    #  colormap hot, hold on, alpha(.8)
    #  set(gca, 'color', 'blue');
    #  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
    #
    # Example (Sinusoidal density):
    #
    # clear all
    #   X=rand(1000,1); Y=sin(X*10*pi)+randn(size(X))/3; data=[X,Y];
    #  % apply routine
    #  [bandwidth,density,X,Y]=kde2d(data);
    #  % plot the data and the density estimate
    #  surf(X,Y,density,'LineStyle','none'), view([0,70])
    #  colormap hot, hold on, alpha(.8)
    #  set(gca, 'color', 'blue');
    #  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
    #
    # Notes: If you have a more accurate density estimator 
    #        (as measured by which routine attains the smallest 
    #         L_2 distance between the estimate and the true density) or you have 
    #        problems running this code, please email me at botev@maths.uq.edu.au 
    #  Reference: Botev, Z. I.,
    #             "A Novel Nonparametric Density Estimator",Technical Report,The University of Queensland
    #             http://espace.library.uq.e
    #             du.au/view.php?pid=UQ:12535
    # global N A2 I

    n=2**np.ceil(np.log2(n)); # round up n to the next power of 2;
    N=data.shape[0]



    

    
    
    def K(s):
        out=(-1)**s * np.prod(np.arange(1,2**s,2)) / np.sqrt(2*np.pi)
        return out

    
    def psi(s, Time, I, A2):
        w=np.exp(-I*np.pi**2*Time) * np.array([1, .5*np.ones( (1, I.__len__() - 1) )])
        wx=w*(I**s[0])
        wy=w*(I**s[1])
        # wx.transpose might need extra dim
        out=(-1)**np.sum(s)*(wy*A2*wx.T)*np.pi**(2*np.sum(s))
        return out
    
    
    def func(s, t, I, A2):
        if np.sum(s)<=4:
            Sum_func=func([s[0]+1,s[1]],t, I, A2)+func([s[0],s[1]+1],t, I, A2)
            time=(-2*K(s[0])*K(s[1])/N/Sum_func)**(1/(2+np.sum(s)))
            out=psi(s, time, I, A2)
        else:
            out=psi(s, t, I, A2)
    
        return out
        
    
    def evolve(t, I, A2):
        Sum_func = func([0,2],t, I, A2) + func([2,0],t, I, A2) + 2*func([1,1],t, I, A2)
        time=(2*np.pi*N*Sum_func)**(-1/3)
        out=(t-time)/time
        return [out,time]

    if (MIN_XY is None) | (MAX_XY is None):
        MIN_XY = data.min(axis = 0)
        MAX_XY = data.max(axis = 0)
        Range = MAX_XY - MIN_XY

        MAX_XY=MAX_XY+Range/4
        MIN_XY=MIN_XY-Range/4
    
    scaling=MAX_XY-MIN_XY

    # if N<=size(data,2)
    #     error('data has to be an N by 2 array where each row represents a two dimensional observation')
    # end

    transformed_data = (data - MIN_XY) / scaling

    initial_data=ndhist(transformed_data,n)
    

    # transformed_data=(data-repmat(MIN_XY,N,1))./repmat(scaling,N,1);
    # %bin the data uniformly using regular grid;
    
    # % discrete cosine transform of initial data
    a= dct2d(initial_data)
    # % now compute the optimal bandwidth^2
    val=np.inf
    t_star=0
    c=0
    I = np.linspace(0,n-1,n)**2
    A2 = a**2
    while np.abs(val)>1e-5:
        val,t_star = evolve(t_star, I, A2)
        c=c+1
        if c>1e3:
            print('Algorithm failed to converge in 1000 iterations')
    
    p_02=func([0,2],t_star, I, A2)
    p_20=func([2,0],t_star, I, A2)
    p_11=func([1,1],t_star, I, A2)
    t_x=(p_02**(3/4)/(4*np.pi*N*p_20**(3/4)*(p_11+(p_20*p_02)**0.5)))**(1/3)
    t_y=(p_20**(3/4)/(4*np.pi*N*p_02**(3/4)*(p_11+(p_20*p_02)**0.5)))**(1/3)
    # % smooth the discrete cosine transform of initial data using t_star
    a_t=np.exp(-np.arange(0,n,1).T**2*np.pi**2*t_y/2)*np.exp(-np.arange(0,n,1)**2*np.pi**2*t_x/2)*a; 
    # %transpose goes with y coord.
    # % now apply the inverse discrete cosine transform
    # if nargout>1
    density=idct2d(a_t)*(numel(a_t)/prod(scaling));
    [X,Y]=meshgrid(MIN_XY(1):scaling(1)/(n-1):MAX_XY(1),MIN_XY(2):scaling(2)/(n-1):MAX_XY(2));
    # end
    bandwidth=sqrt([t_x,t_y]).*scaling;
    end
    
    
    %#######################################
    function data = idct2d(data)
    % computes the 2 dimensional inverse discrete cosine transform
    [nrows,ncols]=size(data);
    % Compute wieghts
    w = exp(i*(0:nrows-1)*pi/(2*nrows)).';
    weights=w(:,ones(1,ncols));
    data=idct1d(idct1d(data)');
        function out=idct1d(x)
            y = real(ifft(weights.*x));
            out = zeros(nrows,ncols);
            out(1:2:nrows,:) = y(1:nrows/2,:);
            out(2:2:nrows,:) = y(nrows:-1:nrows/2+1,:);
        end
    end
    
