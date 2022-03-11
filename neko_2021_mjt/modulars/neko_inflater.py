import torch;
# inflates the feature by trimming blanks.
class neko_inflater:
    def out_attns(this, text_length, A, nB, nH, nW):
        lenText = int(text_length.sum())
        start = 0
        out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)
        for i in range(0, nB):
            cur_length = int(text_length[i])
            out_attns[start: start + cur_length] = A[i, 0:cur_length, :, :]
            start += cur_length
        return out_attns;

    def prob_length(this, out_res, nsteps, nB):
        out_length = torch.zeros(nB)
        for i in range(0, nsteps):
            tens = out_res[i, :, :];
            tmp_result = tens.topk(1)[1].squeeze(-1)
            for j in range(nB):
                if out_length[j].item() == 0 and tmp_result[j] == 0:
                    out_length[j] = i + 1
        for j in range(nB):
            if out_length[j] == 0:
                out_length[j] = nsteps + 1
        return out_length

    def inflate(this,out_emb,out_length):
        nT,nB=out_emb.shape[0],out_emb.shape[1];
        start = 0
        if(out_length is None):
            out_length=this.prob_length(out_emb,nT,nB);
        output = torch.zeros(int(out_length.sum()), out_emb.shape[-1]).type_as(out_emb.data)
        for i in range(0, nB):
            cur_length = int(out_length[i])
            cur_length_=cur_length
            if(cur_length_>nT):
                cur_length_=nT;
            output[start: start + cur_length_] = out_emb[0: cur_length_, i, :]
            # if(scores[cur_length_-1, i, :].argmax().item()!=0):
            #     print("???")
            start += cur_length_
        return output,out_length;
    # def inflate(this,out_emb,out_length):
    #     nT,nB=out_emb.shape[0],out_emb.shape[1];
    #     if (out_length is None):
    #         out_length = this.prob_length(out_emb, nT, nB);
    #     mask=(torch.arange(nT)[None, :] < (out_length[:, None])).reshape(-1).cuda();
    #     return out_emb.permute(1,0,2).reshape(nT*nB,-1)[mask],out_length;
    # def inflate(this,out_emb,out_length):
    #     output1, out_length1=this.inflate1(out_emb,out_length);
    #     output2, out_length2 = this.inflate2(out_emb, out_length);
    #     print((output1-output2).max())
    #     return output1,out_length1;
