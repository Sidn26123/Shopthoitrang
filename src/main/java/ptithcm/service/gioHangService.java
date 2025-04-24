package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.stereotype.Service;

import ptithcm.entity.GioHangEntity;

@Service
@Transactional
public interface gioHangService{
	public List<GioHangEntity> layGioHangCuaUser(int maNd);
	public GioHangEntity layGioHangTheoMaNdVaSanPham(int maNd, String maSp);
	public List<GioHangEntity> layAllGioHang();
	public void addGioHang(GioHangEntity giohang);
	public void updateGioHang(GioHangEntity giohang);
	public void updateSoLuong(int soLuong,int maGh);
	public void deleteGioHang(int maGh);
	void updateSize(String size, int maGh);
	
}