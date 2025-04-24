package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import ptithcm.entity.GioHangEntity;
import ptithcm.dao.gioHangDao;
import ptithcm.dao.gioHangDaoImpl;

@Transactional
@Service
public class gioHangServiceImpl implements gioHangService{

	@Autowired
	gioHangDao gioHangDao;
	
	@Override
	public List<GioHangEntity> layGioHangCuaUser(int maNd) {
		return gioHangDao.layGioHangCuaUser(maNd);
	}

	@Override
	public GioHangEntity layGioHangTheoMaNdVaSanPham(int maNd, String maSp) {
		return gioHangDao.layGioHangTheoMaNdVaSanPham(maNd, maSp);
	}
	
	@Override
	public List<GioHangEntity> layAllGioHang(){
		return gioHangDao.layAllGioHang();
	}

	@Override
	public void addGioHang(GioHangEntity giohang) {
		gioHangDao.addGioHang(giohang);
		
	}

	@Override
	public void updateGioHang(GioHangEntity giohang) {
		gioHangDao.updateGioHang(giohang);
		
	}

	@Override
	public void updateSoLuong(int soLuong, int maGh) {
		gioHangDao.updateSoLuong(soLuong, maGh);
		
	}
	
	@Override
	public void updateSize(String size, int maGh) {
		gioHangDao.updateSize(size, maGh);
		
	}

	@Override
	public void deleteGioHang(int maGh) {
		gioHangDao.deleteGioHang(maGh);
		
	}}