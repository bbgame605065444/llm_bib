# 语音转文本笔记

 Good morning. My name is Novi. I'm here to welcome all of you to this tutorial session.

 So for your information, this tutorial is about plots and metrics for measuring bias,

 fairness, calibration, reliability, and robustness presented by Mark Tiger.

 Mark is a research scientist at Meta. Prior to 2014, he was a professor at NYU and Yale.

 Mark got his PhD at Yale following undergraduate at Princeton.

 Mark likes to stand, which probably could be helpful for this particular tutorial session,

 and he's going to be two and a half hours. So please join me to welcome Mark.

 All right. He's telling the truth that it is two and a half hours long, but we're going to have two breaks.

 Okay. So, you know, you don't need to stay here the entire time.

 And you're welcome to come back after the break, if you like. I hope you will.

 I want to mention a few things about this talk that it's probably a little bit more general than the title actually indicates.

 So there has been a huge focus on fairness and bias measurement in machine learning, as well as measuring sort of reliability or uncertainty quantification.

 But the methods that we're going to see during the tutorial are actually more general than that.

 Hmm. Yeah, what happened?

 Yeah. What happened?

 Oh, thank you.

 All right. Starting again. Yeah. So we're going to go for two and a half hours or two hours, something like this.

 Again, we'll have two breaks. There'll be four parts to talk. We won't have a break for the last two parts, but we'll have a break after the first and after the second.

 Now, normally, there would be sort of Q&A sessions after each part of the talk. We'll do that. But what I'm going to do is allow everybody to go out and circulate and eat or do whatever they want during the breaks and those who want to ask questions to me can come up and do that in person.

 This is a feeling.

 And I'm not going to focus on the question and bias, but those are what we're going to focus on.

 Please ask questions and just whenever you have one, you can come up and ask it at the microphone.

 That also, you know, notice that the every time I got every time I got questions to talk.

 So, you know, how in almost all the machine learning literature, the methods for measuring calibration using reliability diagrams.

 Okay. So this is this is an outline of the talk. We're going to start with calibration and reliability.

 We know what this is.

 Techniques, which use bending, bend responses are bucketing, stratifying whatever you want to call it.

 I mean, it's long, but every time.

 Okay. So this is this is an outline of the talk. We're going to start with calibration and reliability.

 So if you don't know what calibration is, we are going to define it completely and rigorously here.

 So you don't need to know what that is in order to move on. If you do know what it is, it's what you think it is and we are going to treat that.

 This is related to reliability and uncertainty quantification.

 The reliability you'll you'll see in terms of reliability diagrams or, for instance, one very standard way of looking at calibration and we'll discuss those.

 Now those are the standard ways of looking at them. We are going to review how in almost all the machine learning literature.

 The methods for measuring calibration using reliability diagrams, the classic methods are widely criticized and known to have all kinds of unbelievable failure modes.

 And they are what everyone continues to use in machine learning and basically in no other field because in every other field, they have decided that these methods are unacceptable.

 All right. So we're going to see some other alternative methods which don't have these issues, which actually resolve all the issues that continue to be published at infinitum in the in the literature on machine learning for some reason, you know, methods from other fields have been percolated through.

 If you haven't noticed, I didn't originally come from machine learning. I came from applied mathematics and statistics and and physics and so, yeah, I was even on the faculty at Yale Medical School for a little bit.

 And so, you know, we're going to see methods from other areas here today as well.

 I hope that's somewhat useful to you.

 If you want to know more about any of these things, please feel free to ask me, especially during the breaks.

 So we'll start with the classic calibration, which I mentioned was like about certain quantification.

 So that has to do with probabilistic predictions, predicted probabilities, most machine learning models classically, that's what they would do.

 They would predict something and there would be a score associated with it. And you could you could read out that score and see that as sort of an estimate of the probability or some other method for estimating what that probability should be.

 Now, there are going to be two different ways of measuring this. This is a talk on measurement. Right.

 So the first way is using cumulative differences. This is not so classical, even though the methods go back 100 years to come a garb and weener and such.

 So we'll go through those. Those don't have the issues that I mentioned. They work very nicely.

 And then we'll discuss the classic techniques, which use binning bin responses or bucketing stratifying whatever you want to call it.

 If you don't call it any of those things, well, you will after today's tutorial. Those are the standard terms. Bining is probably the most common.

 So the first part of the talk is of interest to everyone, whether or not they care about fairness and bias issues, like discrimination.

 The second part of the talk is more focused on sort of bias and discrimination and this, those sorts of social issues.

 Now, the methods do pertain to more generally than just to discrimination and bias.

 But those are what we're going to focus on here because that's what the community and artificial intelligence is most concerned about right now.

 So that's going to come up with matching scores. If anyone has done propensity score matching or something like this.

 We will discuss all that or matched pair analysis, things like this. That's what the matching has to do.

 And so first we'll look at deviation of a sub population from the full population.

 And that's what we generally recommend in the area of measuring fairness and bias in machine learning and artificial intelligence.

 But then we'll also go and look at the methods that are more general interest for like analyzing randomized controlled trials and in medicine and things like this.

 You know, A, B tests will also qualify under that second heading of deviation directly between two sub populations.

 And I will see some generalizations to controlling for multiple covariates. So if you like mathematical sophistication, we'll be using space filling curves, you know, do the panel and Hilbert and things like this in that section.

 Most of the talk will be quite elementary in terms of the mathematics required. The most sophisticated thing we'll be using for the most part would be a little bit of browning motion.

 And if that's not totally familiar to you, you can review that after your after the talk at your leisure.

 And I'll mention a little bit about random walks and browning motion.

 So we'll see some extensions. So again, there'll be a break after the first part, a break after the second part, both the first and the second part are fairly free standing with the second more, more or less dependent on the first.

 The third part doesn't make any sense if you have an attended the first two. So, so we're going to discuss some generalizations, you know, weighted sampling.

 And most of my work at meta involves effectively every day to sit we have somehow has weights that come along with it.

 In most of the world, unweighted sampling is more common, uniform, sampling just drawing at random, but, you know, that's not what's happening at where I work. So, so we're going to discuss what you need to handle those weighted samples as well.

 You're saying, thinking the whole time during the talk, he's like, all my data sets come with weights, because I work at meta, I work at Google or something, and you're not discussing any of that is any of this relevant. Yes, it will be, but we'll do that at the end.

 Okay. First we'll handle the simplest case where there are no weights in the samples and then, then we'll move on to the more sophisticated case.

 Although, if you read the papers associated with these talks, then you'll see most people treat the, including myself, treat the, treat the weighted case, just right at the get go.

 We'll see that some of the methods are randomized, but you can actually avoid the randomization.

 If you are someone who cringes at the thought of having your algorithm be randomized and give a result that depends on the seed you feed into the pseudo random number generator, well, we'll address your concerns just just at the end.

 And I'll show you some graphical interactive plots as well.

 So here's calibration and reliability.

 We're going to start as I mentioned with probabilistic predictions, you could also call these predicted probabilities.

 And here's what a prediction is. So here's a prediction with a given probability.

 Consider an s is 30% chance of snow. I guess being in Vancouver, maybe I should have changed this to rain, but s goes along with snow. So, so that's what I did.

 Why does it do that?

 The slide is gone.

 I don't know what happened here. Thank you.

 I wish I could promise that wouldn't happen again, but I bet it will.

 Would you like to hazard a guess at the probability?

 Okay, well, maybe this will I was not going to do a live demo, but apparently it's been forced on me.

 So we'll measure whether my prediction of 30% chance is correct.

 Yeah, so, so we're going to look at it just for example here a 30% chance of snow.

 And the reality of course is either one, it's actually going to snow or two, it does not snow.

 And you know, this being a sophisticated audience here, we don't say one in two, we say one in zero, right?

 And then we say, well, we're going to do this for zero properly from computer science.

 Okay, so zero is it did not snow one is it did snow?

 And you say, well, we know what this is, this variable are is just right a Bernoulli random variable, a Bernoulli distribution.

 It takes the expected value of the Bernoulli parameters s.

 And it takes value r as one. The response is one with probability s and r is zero with probability one minus us.

 And so usually when you make an weather predictions, you don't just make it once in your entire lifetime, you probably make several.

 Let's call that n. So we're going to have n independent random results.

 Again, these are responses.

 We'll denote them r one or two up to our n.

 And the prediction is known as perfectly calibrated. Okay, so this is if you didn't know what calibration is, if you didn't know what the talk is about.

 So that prediction is perfectly calibrated. If 30% s is 30% of r one through r n are equal to one.

 And the rest are equal to zero. That means you got the prediction right.

 Yeah, it's calibrated. The probability is what you expect it to be.

 So that's this notation s is going to be the predicted probability, the score.

 I guess I didn't actually mention what they are, but we will shortly and r is the response, what actually happens, what you measure.

 Okay, so s is your prediction, r is what you measure.

 And let me just drive this home.

 You got to memorize this at least for the duration of the talk, probably for your lifetime.

 So we're going to write the predictions with different probabilities and the notation we're using here is the probabilities of success like did it snow is s one s two up through s n.

 These are the probabilities of the expect these probabilities s one through s and are the expected values of Bernoulli distributions.

 And the consequences are one through r n are independent Bernoulli random variables.

 RK and SK are going to come as a pair. So with every prediction you have every s one or every s K in general, you're going to have a corresponding appeared RK response.

 Okay, you can't like to resort them and shuffle them.

 Not in this setup.

 Okay, so the probabilities of success are known as scores and will be viewed as deterministic during this tutorial.

 Some people, especially coming from statistics, like to take the log of their probability before they call it a score.

 You know, the Fisher score is the log probability. The rest of the world usually calls it the exact.

 The probability itself without the log the score. So that's the notation.

 We're going to use here today. So the scores are the probabilities without the logs.

 If you had log, you got to exponentially it before you get the score.

 All right, so the responses are also known as like results or outcomes. There's lots of different.

 They could be like dependent variables. There's lots of other terms for this.

