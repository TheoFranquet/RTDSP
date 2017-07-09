/*************************************************************************************
			       DEPARTMENT OF ELECTRICAL AND ELECTRONIC ENGINEERING
					   		     IMPERIAL COLLEGE LONDON 

 				      EE 3.19: Real Time Digital Signal Processing
					       Dr Paul Mitcheson and Daniel Harvey

				        		 PROJECT: Frame Processing

 				            ********* ENHANCE. C **********
							 Shell for speech enhancement 

  		Demonstrates overlap-add frame processing (interrupt driven) on the DSK. 

 *************************************************************************************
 				             By Danny Harvey: 21 July 2006
							 Updated for use on CCS v4 Sept 2010
 ************************************************************************************/
/*
 *	You should modify the code so that a speech enhancement project is built 
 *  on top of this template.
 */
/**************************** Pre-processor statements ******************************/
//  library required when using calloc
#include <stdlib.h>
//  Included so program can make use of DSP/BIOS configuration tool.  
#include "dsp_bios_cfg.h"

/* The file dsk6713.h must be included in every program that uses the BSL.  This 
   example also includes dsk6713_aic23.h because it uses the 
   AIC23 codec module (audio interface). */
#include "dsk6713.h"
#include "dsk6713_aic23.h"

// math library (trig functions)
#include <math.h>

/* Some functions to help with Complex algebra and FFT. */
#include "cmplx.h"      
#include "fft_functions.h"  

// Some functions to help with writing/reading the audio ports when using interrupts.
#include <helper_functions_ISR.h>

#define WINCONST 0.85185			/* 0.46/0.54 for Hamming window */
#define FSAMP 8000.0				/* sample frequency, ensure this matches Config for AIC */
#define FFTLEN 256					/* fft length = frame length 256/8000 = 32 ms*/
#define NFREQ (1+FFTLEN/2)			/* number of frequency bins from a real FFT */
#define OVERSAMP 4					/* oversampling ratio (2 or 4) */  
#define FRAMEINC (FFTLEN/OVERSAMP)	/* Frame increment */
#define CIRCBUF (FFTLEN+FRAMEINC)	/* length of I/O buffers */
#define OUTGAIN 16000.0				/* Output gain for DAC */
#define INGAIN  (1.0/16000.0)		/* Input gain for ADC  */
#define PI 3.141592653589793		/*PI defined here for use in your code */
#define TFRAME FRAMEINC/FSAMP       /* time between calculation of each frame */


/******************************* Global declarations ********************************/

/* Audio port configuration settings: these values set registers in the AIC23 audio 
   interface to configure it. See TI doc SLWS106D 3-3 to 3-10 for more info. */
DSK6713_AIC23_Config Config = { \
			 /**********************************************************************/
			 /*   REGISTER	            FUNCTION			      SETTINGS         */ 
			 /**********************************************************************/\
    0x0017,  /* 0 LEFTINVOL  Left line input channel volume  0dB                   */\
    0x0017,  /* 1 RIGHTINVOL Right line input channel volume 0dB                   */\
    0x01f9,  /* 2 LEFTHPVOL  Left channel headphone volume   0dB                   */\
    0x01f9,  /* 3 RIGHTHPVOL Right channel headphone volume  0dB                   */\
    0x0011,  /* 4 ANAPATH    Analog audio path control       DAC on, Mic boost 20dB*/\
    0x0000,  /* 5 DIGPATH    Digital audio path control      All Filters off       */\
    0x0000,  /* 6 DPOWERDOWN Power down control              All Hardware on       */\
    0x0043,  /* 7 DIGIF      Digital audio interface format  16 bit                */\
    0x008d,  /* 8 SAMPLERATE Sample rate control        8 KHZ-ensure matches FSAMP */\
    0x0001   /* 9 DIGACT     Digital interface activation    On                    */\
			 /**********************************************************************/
};

DSK6713_AIC23_CodecHandle H_Codec;	/*Codec handle:- a variable used to identify audio interface */

// Inout/Output variables
float *inbuffer, *outbuffer;   	 /* Input/output circular buffers */
float *inframe, *outframe;       /* Input and output frames */
float *inwin, *outwin;  		 /* Input and output windows */
float ingain, outgain;			 /* ADC and DAC gains */ 
float cpufrac; 				     /* Fraction of CPU time used */
volatile int io_ptr=0;           /* Input/ouput pointer for circular buffers */
volatile int frame_ptr=0;        /* Frame pointer */
int io_ptr0; 

// Indexes
int k;
int m;
int frame_count;				/* Number of frame counted*/

// Pointers declarations
complex X_inframe[FFTLEN];		/* Complex equivalent of the new frame*/	
complex Y_outframe[FFTLEN];
float *mag_X;  					/* Compute the Magnitude of X_inframe*/
float *mag_Y;  					/* Compute the Magnitude of X_inframe*/
float *P;						/* Low pass magnitude signal in frequency domain */
float *Q;          				/* Low pass magnitude signal in power domain */
float *M1;						/* Minimum spectrum 1 */
float *M2;						/* Minimum spectrum 2 */
float *M3;						/* Minimum spectrum 3 */
float *M4;						/* Minimum spectrum 4 */
float *tmp;
float *g;						/* Frequency dependant gainb factor*/
float *N;						/* Magnitude Noise in power domain */ 
float *lp_N;					/* Low pass magnitude Noise in  frquency domain */
float *lp_NQ;					/* Low pass magnitude Noise in  power domain */

// Sound parameters
float lambda = 0.01;			/* lambda */
float t_constant = 0.02;	
float t_constant_N = 0.02;		/* time constant */ 
float alpha = 2;				/* alpha */
float treshold=1.5;				/* SNR treshold */ 
float treshold2=7;				/* SNR treshold */ 
float lowSNR_alpha=8;			/* alpha when the SNR is below the treshold */
float highSNR_alpha=1;			/* alpha when the SNR is above the treshold */
float mediumSNR_alpha=4;	
float vol_gain = 3;			/* Output volume gain */

// Computed parameters
float M_duration = OVERSAMP*2.5/(FFTLEN/FSAMP); /* Number of frames for 2.5 secondes*/
float kay;						/* k in the LPF formula */
float kayN;
float q;						/* k in the LPF formula */
float qN;
float NSR;						/* Noise to signal ratio*/
float SNR;						/* Signal to noise ratio*/
float NPR;						/* Noise to low pass signal ratio*/
float PSR;						/* Low pass signal to signal ratio*/

// Switches
int enhance=1;					/* Switch between with and without processing */ 
int type=10;					/* Switch between different enhancement types */ 
int gfunction=0;				/* Switch between different functions of g(w) */ 
int oversubtraction=1;			/* Switch between with or without oversubtraction */ 

 /******************************* Function prototypes *******************************/
void init_hardware(void);    	/* Initialize codec */ 
void init_HWI(void);            /* Initialize hardware interrupts */
void ISR_AIC(void);             /* Interrupt service routine for codec */
void process_frame(void);       /* Frame processing routine */
float min(float a,float b);
float max(float a,float b);

/********************************** Main routine ************************************/
void main()
{      

	/*  Initialize and zero fill arrays */  
	inbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Input array */
    outbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Output array */
	inframe		= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    outframe	= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    inwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Input window */
    outwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Output window */
	mag_X		= (float *) calloc(FFTLEN, sizeof(float));
	mag_Y		= (float *) calloc(FFTLEN, sizeof(float));
	P			= (float *) calloc(FFTLEN, sizeof(float));	
	Q			= (float *) calloc(FFTLEN, sizeof(float));	
	M1			= (float *) calloc(FFTLEN, sizeof(float));
	M2			= (float *) calloc(FFTLEN, sizeof(float));
	M3			= (float *) calloc(FFTLEN, sizeof(float));
	M4			= (float *) calloc(FFTLEN, sizeof(float));
	tmp			= (float *) calloc(FFTLEN, sizeof(float));
	g			= (float *) calloc(FFTLEN, sizeof(float));
	N			= (float *) calloc(FFTLEN, sizeof(float));
	lp_N		= (float *) calloc(FFTLEN, sizeof(float));
	lp_NQ		= (float *) calloc(FFTLEN, sizeof(float));
	
	/* initialize board and the audio port */
  	init_hardware();
  
  	/* initialize hardware interrupts */
  	init_HWI();    
  
	/* initialize algorithm constants */  
                       
  	for (k=0;k<FFTLEN;k++)
	{                           
	inwin[k] = sqrt((1.0-WINCONST*cos(PI*(2*k+1)/FFTLEN))/OVERSAMP);
	outwin[k] = inwin[k]; 
	} 
  	ingain=INGAIN;
  	outgain=OUTGAIN;     
  	   
	/* initialize the frame count */  
 	frame_count = 1;	
 					
  	/* main loop, wait for interrupt */  
  	while(1){
  	q=exp(-TFRAME/t_constant);
  	qN = exp(-TFRAME/t_constant_N);
  	process_frame();
  	}
}
    
/********************************** init_hardware() *********************************/  
void init_hardware()
{
    // Initialize the board support library, must be called first 
    DSK6713_init();
    
    // Start the AIC23 codec using the settings defined above in config 
    H_Codec = DSK6713_AIC23_openCodec(0, &Config);

	/* Function below sets the number of bits in word used by MSBSP (serial port) for 
	receives from AIC23 (audio port). We are using a 32 bit packet containing two 
	16 bit numbers hence 32BIT is set for  receive */
	MCBSP_FSETS(RCR1, RWDLEN1, 32BIT);	

	/* Configures interrupt to activate on each consecutive available 32 bits 
	from Audio port hence an interrupt is generated for each L & R sample pair */	
	MCBSP_FSETS(SPCR1, RINTM, FRM);

	/* These commands do the same thing as above but applied to data transfers to the 
	audio port */
	MCBSP_FSETS(XCR1, XWDLEN1, 32BIT);	
	MCBSP_FSETS(SPCR1, XINTM, FRM);	
	

}
/********************************** init_HWI() **************************************/ 
void init_HWI(void)
{
	IRQ_globalDisable();			// Globally disables interrupts
	IRQ_nmiEnable();				// Enables the NMI interrupt (used by the debugger)
	IRQ_map(IRQ_EVT_RINT1,4);		// Maps an event to a physical interrupt
	IRQ_enable(IRQ_EVT_RINT1);		// Enables the event
	IRQ_globalEnable();				// Globally enables interrupts

}
        
/******************************** process_frame() ***********************************/  
void process_frame(void)
{
	/* work out fraction of available CPU time used by algorithm */    
	cpufrac = ((float) (io_ptr & (FRAMEINC - 1)))/FRAMEINC;  
		
	/* wait until io_ptr is at the start of the current frame */ 	
	while((io_ptr/FRAMEINC) != frame_ptr); 
	
	/* then increment the framecount (wrapping if required) */ 
	if (++frame_ptr >= (CIRCBUF/FRAMEINC)) frame_ptr=0;
 	
 	/* save a pointer to the position in the I/O buffers (inbuffer/outbuffer) where the 
 	data should be read (inbuffer) and saved (outbuffer) for the purpose of processing */
 	io_ptr0=frame_ptr * FRAMEINC;
	
	/* copy input data from inbuffer into inframe (starting from the pointer position) */ 
	 
	m=io_ptr0;
    for (k=0;k<FFTLEN;k++)
	{                           
		inframe[k] = inbuffer[m] * inwin[k]; 
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	} 
	
/*************************** DO PROCESSING OF FRAME  HERE ***************************/
	
	// Switch between processing or not of the incoming signal
	switch (enhance){
		
//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// Without Processing /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 0: 
		for (k=0;k<FFTLEN;k++) {                           
			outframe[k] = inframe[k]; /* copy input straight into output */ 
		}
		break;
		
//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Processing /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 1: 
		
		// Preprocessing common to all enhancement 		

		// Every 2.5s or 312 frames counted
		if (frame_count > M_duration){
			
			// compute the Noise lvl N(w) and amplify
			for (k=0; k<FFTLEN; k++){
				// switch between oversubtraction or usual noise calculating method
				switch(oversubtraction){
					// Compute the noise with oversubtraction, enhancement 6
					case 1: 
						N[k] = min(min(M1[k],M2[k]),min(M3[k],M4[k]));
						// Calculate the signal to noise ratio		
						SNR = mag_X[k]/N[k];
						// Operate the oversubtraction with three alpha levels
						if(SNR<=treshold){
							// for low SNR increase attenuation
							N[k]   = lowSNR_alpha*N[k];
						} 
						else if(treshold<SNR<=treshold2){
							N[k]   = mediumSNR_alpha*N[k];
						}
						else if(SNR>treshold2){
							// for high SNR keep an identical the signal 
							N[k]   = highSNR_alpha*N[k];
						}
					break;
					
					// Compute the noise without oversubtraction
					case 0:
						N[k] = alpha*min(min(M1[k],M2[k]),min(M3[k],M4[k]));
					break;
				}
							
				// Low pass filter the noise in the power domain
				lp_NQ[k] = (1-qN)*(N[k]*N[k])+qN*(lp_N[k]*lp_N[k]);
				lp_N[k] = sqrt(lp_NQ[k]);
			}	
			
			// rotate the buffers and restart the frame count
			M4=M3;
			M3=M2;
			M2=M1;
			frame_count = 1;
		}
		
		// preprocess input frame, inframe, into complex form
		for (k=0; k<FFTLEN; k++){
			X_inframe[k].r = inframe[k];
			X_inframe[k].i = 0;
		}
		
		// Compute the FFT of the complex form of the frame
		// X_inframe is transformed from time domain to frequency
		fft(FFTLEN, X_inframe);
		
		// Calculate the magnitude of the frequency signal for all frequnecy bins 
		for (k=0; k<FFTLEN; k++){
			mag_X[k] = cabs(X_inframe[k]);
			
		// initialize to first frame amplitudes when we calculate a new value of M1
		if (frame_count == 1){
				M1[k] = mag_X[k];
			}
		}
		
		// Determine which enhancement we are using
		switch (type){	
	
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CASE 0 ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

		case 0: // simple processing without any further enhancement 
		
		// find minimum for a single frame
		for (k=0; k<FFTLEN; k++){ 
			if (mag_X[k]<=M1[k]){
				M1[k] = mag_X[k];
			}
		}
		
		// compute g(w)
		for (k=0; k<FFTLEN; k++){
			NSR = (N[k]/mag_X[k]);
			if (lambda>(1-NSR)){
				g[k] = lambda;
			}
			else{
				g[k] = (1-NSR);
			}
		}
		
		// supress noise and compute output
		for (k=0; k<FFTLEN; k++){
			X_inframe[k].r*=g[k];
			X_inframe[k].i*=g[k];
		}
		
		// IFFT 
		// transform the output from the frequency to the time domain 
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}
		
		// count up to the number of frames in 2.496s (312 frame)
		frame_count++;
		
		break;
		
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CASE 1 ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 1: // simple processing w/ LowPass filter on the input frame
				
		// create a low pass P of the inframe
		for (k=1; k<FFTLEN; k++){
			// low pass filter magnitude domain
			P[k] = (1-q)*mag_X[k]+q*P[k-1];
			if (P[k]<=M1[k]){
				M1[k] = P[k];
			}
		}
		
		// compute g(w)
		for (k=0; k<FFTLEN; k++){
			NSR = (N[k]/mag_X[k]);
			if (lambda>(1-NSR)){
				g[k] = lambda;
			}
			else{
				g[k] = (1-NSR);
			}
		}
		
		// supress noise and compute output
		for (k=0; k<FFTLEN; k++){
			X_inframe[k].r*=g[k];
			X_inframe[k].i*=g[k];
		}
		
		// IFFT 
		// transform the output from the frequency to the time domain 
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}
		
		// count up to the number of frames in 2.496s (312 frame)
		frame_count++;
		
		break;
		
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CASE 2 ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 2: // simple processing w/ LowPass filter in the power domain on the input
		
		// create a low pass P using the inframe
		for (k=1; k<FFTLEN; k++){
			Q[k] = (1-q)*(mag_X[k]*mag_X[k])+q*Q[k-1];
			P[k] = sqrt(Q[k]);
			// find minimum for a single frame
			if (P[k]<=M1[k]){
				M1[k] = P[k];
			}
		}

		// compute g(w)
		for (k=0; k<FFTLEN; k++){
			NSR = (N[k]/mag_X[k]);
			if (lambda>(1-NSR)){
				g[k] = lambda;
			}
			else{
				g[k] = (1-NSR);
			}
		}
		
		// supress noise and compute output
		for (k=0; k<FFTLEN; k++){
			X_inframe[k].r*=g[k];
			X_inframe[k].i*=g[k];
		}
		
		// IFFT 
		// transform the output from the frequency to the time domain 
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}
		
		// count up to the number of frames in 2.496s (312 frame)
		frame_count++;
		
		break;
	
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CASE 3 ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 3: // simple processing w/ LowPass filtering input and noise in power domain
				
		// create a low pass P using the inframe
		for (k=1; k<FFTLEN; k++){
			Q[k] = (1-q)*(mag_X[k]*mag_X[k])+q*Q[k-1];
			P[k] = sqrt(Q[k]);
		// find minimum for a single frame
			if (P[k]<=M1[k]){
				M1[k] = P[k];
			}
		
		// compute g(w)
		for (k=0; k<FFTLEN; k++){
			NSR = (N[k]/mag_X[k]);
			if (lambda>(1-NSR)){
				g[k] = lambda;
			}
			else{
				g[k] = (1-NSR);
			}
		}
		
		// supress noise and compute output
		for (k=0; k<FFTLEN; k++){
			X_inframe[k].r*=g[k];
			X_inframe[k].i*=g[k];
		}
		
		// IFFT 
		// transform the output from the frequency to the time domain 
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}
		
		// count up to the number of frames in 2.496s (312 frame)
		frame_count++;
		
		break;
		
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CASE 4 ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 4: // processing w/ different g functions
		
		// create a low pass P in the power domain of the complex inframe
		for (k=0; k<FFTLEN; k++){
			Q[k] = (1-q)*(mag_X[k]*mag_X[k])+q*Q[k-1];
			P[k] = sqrt(Q[k]);
			// find minimum for a single frame
			if (P[k]<=M1[k]){
				M1[k] = P[k];
			}
		}
					
		switch (gfunction){
		
		case 1:
		// enhancement 5
			for (k=0; k<FFTLEN; k++){
				// compute g(w)			
				NSR = (lp_N[k]/mag_X[k]);
				if (lambda>(1-NSR)){
					g[k] = lambda;
				}
				else{
					g[k] = (1-NSR);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 2:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				NSR = (lp_N[k]/mag_X[k]);
				if (lambda*NSR>(1-NSR)){
					g[k] = lambda*NSR;
				}
				else{
					g[k] = 1-NSR;
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 3:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				PSR = (P[k]/mag_X[k]);
				if (lambda*PSR>1-(lp_N[k]/mag_X[k])){
					g[k] = lambda*PSR;
				}
				else{
					g[k] = 1-(lp_N[k]/mag_X[k]);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 4:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				NPR = (lp_N[k]/P[k]);
				if (lambda*NPR>(1-NPR)){
					g[k] = lambda*NPR;
				}
				else{
					g[k] = 1-NPR;
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 5:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				NPR = (lp_N[k]/P[k]);
				if (lambda>(1-NPR)){
					g[k] = lambda;
				}
				else{
					g[k] = 1-NPR;
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		}
		
		// IFFT 
		// transform the output from the frequency to the time domain 
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}
		
		// count up to the number of frames in 2.496s (312 frame)
		frame_count++;
		
		break;
		
		
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// CASE 5 ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
		
		case 5: // processing w/ different g functions in the power domain
		
		// create a low pass P in the power domain of the complex inframe
		for (k=0; k<FFTLEN; k++){
			Q[k] = (1-q)*(mag_X[k]*mag_X[k])+q*Q[k-1];
			P[k] = sqrt(Q[k]);
			// find minimum for a single frame
			if (P[k]<=M1[k]){
				M1[k] = P[k];
			}
		}
					
		switch (gfunction){
		
		case 1:
		// enhancement 5
			for (k=0; k<FFTLEN; k++){
				// compute g(w)			
				NSR = (lp_N[k]/mag_X[k]);
				NSR = NSR*NSR;
				if (lambda>sqrt(1-NSR)){
					g[k] = lambda;
				}
				else{
					g[k] = sqrt(1-NSR);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 2:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				NSR = (lp_N[k]/mag_X[k]);
				NSR = NSR*NSR;
				if (lambda*NSR>(1-NSR)){
					g[k] = lambda*NSR;
				}
				else{
					g[k] = sqrt(1-NSR);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 3:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				PSR = (P[k]/mag_X[k]);
				NSR = (lp_N[k]/mag_X[k]);
				NSR = NSR*NSR;
				if (lambda*PSR>(1-NSR)){
					g[k] = lambda*PSR;
				}
				else{
					g[k] = sqrt(1-NSR);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 4:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				NPR = (lp_N[k]/P[k]);
				NPR = NPR*NPR;
				if (lambda*NPR>sqrt(1-NPR)){
					g[k] = lambda*NPR;
				}
				else{
					g[k] = sqrt(1-NPR);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		
		case 5:
			for (k=0; k<FFTLEN; k++){
				// compute g(w)
				NPR = (lp_N[k]/P[k]);
				NPR = NPR*NPR;
				if (lambda>sqrt(1-NPR)){
					g[k] = lambda;
				}
				else{
					g[k] = sqrt(1-NPR);
				}
				// supress noise
				X_inframe[k].r*=g[k];
				X_inframe[k].i*=g[k];
			}
		break;
		}
		
		// IFFT 
		// transform the output from the frequency to the time domain 
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}
		
		// count up to the number of frames in 2.496s (312 frame)
		frame_count++;
		
		break;
		
		
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// BEST CASE //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
			
		case 10:
		
		// create a low pass P in the power domain of the complex inframe
		for (k=0; k<FFTLEN; k++){
			// low pass filter in the Power domain
			Q[k] = (1-q)*(mag_X[k]*mag_X[k])+q*(P[k]*P[k]);
			P[k] = sqrt(Q[k]);
			// find minimum for a single frame
			if (P[k]<=M1[k]){
				M1[k] = P[k];
			}
			// compute g(w) and calculate Y
			NSR = lp_N[k]/mag_X[k];
			PSR = (P[k]/mag_X[k]);
			if (lambda*PSR>1-NSR){
				g[k] = lambda*PSR;
			}
			else if (lambda*PSR<=1-NSR){
				g[k]=(1-NSR); 
			}
			
			X_inframe[k].r*=vol_gain*g[k];
			X_inframe[k].i*=vol_gain*g[k];
		}
		
		// IFFT
		ifft(FFTLEN, X_inframe);
		
		// postprocess to output
		for (k=0; k<FFTLEN; k++){
			 outframe[k] = X_inframe[k].r;
		}

		// count up to the number of frames in 2.5s
		frame_count++;

		break;
		}
	break;
	}
/**********************************************************************************/
	
    // multiply outframe by output window and overlap-add into output buffer 
                           
	m=io_ptr0;
    
    for (k=0;k<(FFTLEN-FRAMEINC);k++){   	/* this loop adds into outbuffer */                       
	  	outbuffer[m] = outbuffer[m]+outframe[k]*outwin[k];   
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	}         
	
    for (;k<FFTLEN;k++){                           
		outbuffer[m] = outframe[k]*outwin[k];   /* this loop over-writes outbuffer */        
	    m++;
	}	                                   
}        
/*************************** INTERRUPT SERVICE ROUTINE  *****************************/

// Map this to the appropriate interrupt in the CDB file

void ISR_AIC(void)
{       
	short sample;
	/* Read and write the ADC and DAC using inbuffer and outbuffer */
	
	sample = mono_read_16Bit();
	inbuffer[io_ptr] = ((float)sample)*ingain;
		/* write new output data */
	mono_write_16Bit((int)(outbuffer[io_ptr]*outgain)); 
	
	/* update io_ptr and check for buffer wraparound */    
	
	if (++io_ptr >= CIRCBUF) io_ptr=0;
}

/************************************* FUNCTIONS USED *****************************************/

float min(float a,float b){
	if (a>b){
		return b;
	}
	else{
		return a;
	}
}

float max(float a,float b){
	if (a<b){
		return b;
	}
	else{
		return a;
	}
}
