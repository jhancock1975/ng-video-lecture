def main():
    
    torch.manual_seed(1337)

    encode, decode, vocab_size, data = process_input_file('input.txt')


    
    # Train and test splits
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
            
    gpt_params = GPTParams()
    model = GPTLanguageModel(gpt_params, vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpt_params.learning_rate)
    
    for iter in range(gpt_params.max_iters):
    
        # every once in a while evaluate the loss on train and val sets
        if iter % gpt_params.eval_interval == 0 or iter == gpt_params.max_iters - 1:
            losses = estimate_loss(gpt_params, model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Save the checkpoint
            checkpoint = {
                'epoch': iter // gpt_params.eval_interval,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, 'checkpoint.pt')
    
    
        # sample a batch of data
        xb, yb = get_batch('train', gpt_params, train_data, val_data)
    
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500, gpt_params=gpt_params)[0].tolist()))
    open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000, gpt_params=gpt_params)[0].tolist()))

if __name__ == '__main__':
    main()
