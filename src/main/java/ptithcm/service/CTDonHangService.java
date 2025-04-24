package ptithcm.service;



import java.util.List;

import ptithcm.entity.CTDonHangEntity;


public interface CTDonHangService{
	public void luuCtdh(CTDonHangEntity ctdh) ;
	public void updateCtdh(CTDonHangEntity ctdh) ;
	public List<CTDonHangEntity> timctdhTheoMaDh(int maDh);
    public CTDonHangEntity timCtdhTheoMaCtdh(int maCTDH);
	public CTDonHangEntity timCtdhTheoMaDhMaSP(int maDHTypeInt, String maSp);
}