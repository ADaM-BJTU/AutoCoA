<div align="center">
<h1>Agent models: Internalizing Chain-of-Action Generation into Reasoning models</h1>
<a href="https://github.com/ADaM-BJTU/AutoCoA/blob/main/CoA_paper.pdf" target="_blank">
    <img src="https://img.shields.io/badge/PDF-Download-red?logo=adobeacrobatreader" alt="PDF">
</a>
<a href="https://arxiv.org/abs/2503.06580" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv">
</a>
</div>

AutoCoA (Automatic generation of Chain-of-Action) is an agent model framework that enhances the multi-turn tool usage capability of reasoning models. The framework internalizes the Chain-of-Action (CoA) generation, allowing agent models to autonomously decide when and how to use external tools to improve task completion in open-domain QA tasks, particularly those requiring long-term reasoning and multi-step actions.


## Features
- **End-to-end Tuning:** Combines Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to enable seamless transition between reasoning and action.
- **Step-level Action Triggering:** Teaches the model when to take action using contrastive learning at critical reasoning junctures.
- **Trajectory-level CoA Optimization:** Enhances the model's behavior through full CoT (Chain-of-Thought) and CoA sequences.
- **Internal World Model:** Reduces interaction costs with real environments by simulating tool interactions during training.

## 📰 News
- 2025.03.09 — Code Released
- 2025.03.09 — Paper Released


## TODO
- Exploring more effective methods for internal world modeling or simulated environment usage.
- RFT in Tool Usage

## Input Output Example
```
<｜User｜>You are an intelligent reasoning assistant with access to a web search tool to help you obtain accurate information. When answering the user's questions, please adhere to the following guidelines:

1. **Uncertainty Handling:**  
   - Encourage the use of web search to solve the problems mentioned during reasoning. 
   - If you are uncertain or do not have sufficient knowledge about a specific detail, you **must** perform a web search. 
   - To perform a web search the format **must** be in the format mentioned in point 2 Web Search Format
   - You can perform multiple web searches to ensure that the answer is correct.
   - The web search should be initiated only within your internal reasoning (the "thinking" phase), and it should not appear in your final answer to the user.

2. **Web Search Format:**  
   - When calling the web search tool, use the following exact format:
     ```
     <begin_search> your query here </end_search>
     ```
   - After you perform a web search, you will receive results enclosed in:
     ```
     <search_result> ... </search_result>
     ```
   - You can then use this information to further refine your reasoning and answer.

3. **Process:**  
   - If you encounter any uncertainty or unknown knowledge, embed a web search query within your internal thoughts (surrounded by `<think></think>` tags) using the specified format.
   - Incorporate the search results into your reasoning before finalizing your answer.
   - You should provide your final answer in the format \boxed{YOUR_ANSWER}. 

Now, please answer the user's question below: 
If it Feels Good Do it is featured in what 2004 film starring Emile Hirsch and Elisha Cuthbert?

<｜Assistant｜><think>
Okay, so I need to figure out which 2004 film starring Emile Hirsch and Elisha Cuthbert features the song "If It Feels Good Do It." Hmm, I'm not entirely sure about this, but I'll try to work through it step by step.

First, I know that Emile Hirsch and Elisha Cuthbert were part of the band The Offspring. They were pretty popular in the late '90s and early 2000s. I remember they had some hits like "Dookie" and "Candy-Coated Love." But I'm not sure if "If It Feels Good Do It" was one of their songs used in a movie.

Wait, maybe it was used in a movie soundtrack? I think some bands' songs get featured in movies, especially if the movie is based on their music. Let me think about movies from 2004 that had any connection to The Offspring.

I recall that there was a movie called "The Last Days of Disco," which came out in 2004. It was about the end of the disco era, and I think it had some music from The Offspring. But I'm not certain if "If It Feels Good Do It" was in that movie.

Alternatively, maybe it was in another movie. Let me try to remember other 2004 films with those actors. There's "The Room," but that's more of a drama and doesn't seem related to music. Then there's "Shrek 2," but that's animated and doesn't fit either.

Wait, perhaps it's a lesser-known film. Maybe I should look up which 2004 movie stars Emile Hirsch and Elisha Cuthbert and see if it includes that song. Alternatively, I could check if "If It Feels Good Do It" was used in any 2004 movies.

I'm a bit stuck here, so I think I should perform a web search to get accurate information. Let me search for "If It Feels Good Do It" and see which movie it's in.

[<begin_search>If It Feels Good Do It movie</end_search>

<search_result> result 1: "If It Feels Good Do It"
"If It Feels Good Do It" is a song by Canadian rock band Sloan. It was released as the lead single from the band's sixth studio album, "Pretty Together". The song was first made available online in August 2001. The song was nominated for Best Single at the 2002 Juno Awards. The song is featured on the soundtrack of the EA Sports video game Triple Play 2002 and is also featured in the 2004 film "The Girl Next Door". Music video. The music video for "If It Feels Good Do It" was directed by Noble Jones. There are two different versions of the music video. The original version, known as the "People Version", which aired in Canada features a transition between the band playing and commentary of several people expressing their opinion on what rock and roll is about (in reference to Chris Murphy's opening line of the song). This version also features a cameo by Canadian rock musician Danko Jones. The other version, known as the "Performance Version", which aired in the U.S. features just the band performing without the commentary. The video won the award for "Best Video" at the East Coast Music Awards. The video also won the award for "Best Cinematography" at the 2002
result 2: "If It Feels Good Do It"
MuchMusic Video Awards, while peaking at #6 on MuchMusic's Top 30 Countdown.
result 3: "Feels Good"
"Days of Our Lives", and in the 2007 movie "Trick 'r Treat". </search_result>



Hmm, according to the first search result, "If It Feels Good Do It" is a song by Sloan, and it's featured in the 2004 film "The Girl Next Door." Wait, but the user mentioned Emile Hirsch and Elisha Cuthbert. I thought they were in "The Last Days of Disco," but maybe I'm confusing the actors.

Let me check who stars in "The Girl Next Door." I think it's a different cast. Maybe I should look up the cast of "The Girl Next Door" to see if it includes Emile Hirsch and Elisha Cuthbert.

[<begin_search>Cast of The Girl Next Door</end_search>

<search_result> result 1: "The Girl Next Door (2004 film)"
The Girl Next Door is a 2004 American romantic comedy film about a high school senior who falls in love for the first time with the girl next door, but finds the situation becoming complicated after he learns that she is a former pornographic actress. It stars Emile Hirsch, Elisha Cuthbert, Timothy Olyphant, James Remar, Chris Marquette and Paul Dano and is directed by Luke Greenfield. Despite mixed reviews and low theatrical attendance at the time it has gained the status of a cult classic over the years. Plot. Ambitious high school senior Matthew Kidman has been accepted to Georgetown University, but cannot afford the tuition. He has raised $25,000 in order to bring a brilliant Cambodian student, Samnang, to study in the United States, but finds little else truly memorable about his high school experience. His life suddenly changes when Danielle moves in next door. Matthew witnesses her undressing from his bedroom window, until she sees him and storms over, knocking on the door and introducing herself to his parents. They suggest to Matthew that he show Danielle around town. While driving around, Danielle stops the car and forces Matthew to get out and strip for her. The two get to know each other through weird adventures, which include Matthew finding
result 2: "The Woman Next Door (1919 film)"
The Woman Next Door is a 1919 American silent drama film directed by Robert G. Vignola and written by Carolyn Wells. The film stars Ethel Clayton, Emory Johnson, Noah Beery, Sr., Jane Wolfe, Katherine Griffith, and Genevieve Blinn. The film was released on May 18, 1919, by Paramount Pictures. Plot. A man mistakes another man's wife as that man's daughter and starts a love affair with her. Production. The working titles for the film were The Girl Next Door and Vicky.  Preservation status. No prints exist of this film.
result 3: "Falling in Love with the Girl Next Door"
Falling in Love With the Girl Next Door is a 2006 made-for-television comedy film recorded primarily on location in Avalon, California. It premiered on Hallmark Channel on February 4, 2006, and pits Shelley Long and Patty Duke as feuding neighbors. Plot. Bridget Connolly (Patty Duke) and Betsy Lucas (Shelley Long) are two social climbing society women from Catalina Island who have been continuous rivals for some unknown reason. But when Betsy's son Mark falls in love with the girl next door—who happens to be Bridget's daughter Theresa—the two women put all of their energy in to planning the wedding, much to the chagrin of their children. Production notes. Bruce Boxleitner and Patrick Duffy bonded while filming, playing golf regularly when they weren't shooting the film. Long, Duke, Duffy, and Boxleitner also had fun off the set together. Almost every night after filming, the four adults would run up and down the hotel hallways and lobby armed with cap pistols and have shoot-outs. Patrick Duffy explains: Adds Boxleitner: "So every night in Patty's room, on this little coffee table that was so small, Patrick and I were like gangster gamblers and the girls were </search_result>



Okay, so the first result confirms that "The Girl Next Door" is the 2004 film starring Emile Hirsch and Elisha Cuthbert. And according to the initial search, the song "If It Feels Good Do It" is featured in that movie. So putting it together, the song is in "The Girl Next Door."

I think that answers the question. The film is "The Girl Next Door," and it stars Emile Hirsch and Elisha Cuthbert, featuring the song "If It Feels Good Do It."
</think>

The 2004 film featuring "If It Feels Good Do It" and starring Emile Hirsch and Elisha Cuthbert is "The Girl Next Door."

\boxed{The Girl Next Door}
```


## Acknowledge

This implementation draws heavily on [verl](https://github.com/volcengine/verl)  and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). We extend our sincere gratitude to these projects for their valuable contributions to the open-source community.

## License

This work is released under the MIT License. See the [LICENSE](./LICENSE) file for more details. By using this code or associated materials, you agree to comply with the terms outlined in the license.



<!--
****
## Citation

If this work is helpful to your research, please cite our paper:

<!-- ```
@article{
to 
}
``` -->
